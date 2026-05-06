#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <emscripten.h>
#include <emscripten/html5.h>
#include <GLES3/gl3.h>
#include <SDL2/SDL.h>

#include "imgui.h"
#include "imgui_impl_opengl3.h"

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Mat4 {
    float m[16]{};
};

struct Vertex {
    Vec3 position; // Reference capsule parameter position.
    Vec3 normal;   // Reference normal; retained to show the shared mesh contract.
    float vAlong;  // 0 at bottom, 1 at top.
    float region;  // 0 bottom cap, 1 body, 2 top cap.
    float bodyT;   // 0..1 along the reference body.
};

struct Instance {
    Vec3 position;
    float height; // Center-to-center distance h between the two source spheres.
    float r1;
    float r2;
    Vec3 color;
};

struct RadiusControls {
    float r1Min = 0.08f;
    float r1Max = 0.68f;
    float r2Min = 0.08f;
    float r2Max = 0.68f;
};

struct DemoControls {
    int gridCount = 10; // N: the grid is N x N capsules.
    RadiusControls radii;
};

static constexpr float kPi = 3.14159265358979323846f;
static constexpr float kSpacing = 2.55f;
static constexpr float kSphereCenterDistance = 2.4f;

static SDL_Window* gWindow = nullptr;
static SDL_GLContext gGlContext = nullptr;
static GLuint gCapsuleProgram = 0;
static GLuint gGridProgram = 0;
static GLuint gCapsuleVao = 0;
static GLuint gCapsuleVertexBuffer = 0;
static GLuint gCapsuleIndexBuffer = 0;
static GLuint gInstanceBuffer = 0;
static GLuint gGridVao = 0;
static GLuint gGridBuffer = 0;
static GLint gCapsuleViewProjLocation = -1;
static GLint gCapsuleCameraPosLocation = -1;
static GLint gGridViewProjLocation = -1;
static uint32_t gWidth = 1280;
static uint32_t gHeight = 900;
static uint32_t gIndexCount = 0;
static uint32_t gInstanceCount = 0;
static uint32_t gGridVertexCount = 0;
static GLint gReportedSamples = 0;

static DemoControls gControls;
static float gYaw = 38.0f * kPi / 180.0f;
static float gPitch = 28.0f * kPi / 180.0f;
static float gDistance = 27.0f;
static Vec3 gCameraTarget{0.0f, kSphereCenterDistance * 0.55f, 0.0f};
static Vec3 gCameraPos{0.0f, 0.0f, 0.0f};
static Mat4 gViewProj;
static float gHalfGridExtent = 0.0f;
static float gRunningFrameMs = 0.0f;
static bool gResetPressed = false;

static std::vector<Vertex> gVertices;
static std::vector<uint32_t> gIndices;
static std::vector<Instance> gInstances;
static std::vector<Vec3> gGridLines;

static float radians(float degrees) {
    return degrees * kPi / 180.0f;
}

static float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

static Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vec3 operator*(Vec3 v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

static Vec3& operator+=(Vec3& a, Vec3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

static float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

static Vec3 normalize(Vec3 v) {
    const float len = std::sqrt(dot(v, v));
    if (len <= 0.000001f) {
        return {0.0f, 1.0f, 0.0f};
    }
    return {v.x / len, v.y / len, v.z / len};
}

static Mat4 identity() {
    Mat4 r{};
    r.m[0] = 1.0f;
    r.m[5] = 1.0f;
    r.m[10] = 1.0f;
    r.m[15] = 1.0f;
    return r;
}

static Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            r.m[col * 4 + row] =
                a.m[0 * 4 + row] * b.m[col * 4 + 0] +
                a.m[1 * 4 + row] * b.m[col * 4 + 1] +
                a.m[2 * 4 + row] * b.m[col * 4 + 2] +
                a.m[3 * 4 + row] * b.m[col * 4 + 3];
        }
    }
    return r;
}

static Mat4 perspectiveOpenGL(float fovyRadians, float aspect, float nearPlane, float farPlane) {
    Mat4 r{};
    const float f = 1.0f / std::tan(fovyRadians * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
    return r;
}

static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    const Vec3 f = normalize(center - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);

    Mat4 r = identity();
    r.m[0] = s.x;
    r.m[4] = s.y;
    r.m[8] = s.z;
    r.m[1] = u.x;
    r.m[5] = u.y;
    r.m[9] = u.z;
    r.m[2] = -f.x;
    r.m[6] = -f.y;
    r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

static void addRing(std::vector<Vertex>& vertices,
                    int segments,
                    float y,
                    float ringRadius,
                    Vec3 normal,
                    float vAlong,
                    float region,
                    float bodyT) {
    for (int s = 0; s < segments; ++s) {
        const float a = 2.0f * kPi * static_cast<float>(s) / static_cast<float>(segments);
        const float ca = std::cos(a);
        const float sa = std::sin(a);
        vertices.push_back({
            {ringRadius * ca, y, ringRadius * sa},
            normalize({normal.x * ca, normal.y, normal.z * sa}),
            vAlong,
            region,
            bodyT,
        });
    }
}

static void connectRings(std::vector<uint32_t>& indices, int segments, int ringA, int ringB) {
    const uint32_t baseA = static_cast<uint32_t>(ringA * segments);
    const uint32_t baseB = static_cast<uint32_t>(ringB * segments);
    for (int s = 0; s < segments; ++s) {
        const uint32_t a0 = baseA + static_cast<uint32_t>(s);
        const uint32_t a1 = baseA + static_cast<uint32_t>((s + 1) % segments);
        const uint32_t b0 = baseB + static_cast<uint32_t>(s);
        const uint32_t b1 = baseB + static_cast<uint32_t>((s + 1) % segments);
        indices.push_back(a0);
        indices.push_back(b0);
        indices.push_back(b1);
        indices.push_back(a0);
        indices.push_back(b1);
        indices.push_back(a1);
    }
}

static void createSharedUnitCapsuleMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    // This is the only capsule mesh built on the CPU. It is just a reusable
    // parameter mesh; the WebGL2 vertex shader maps each ring to the per-instance
    // convex hull defined by h, r1, and r2.
    constexpr int segments = 32;
    constexpr int capStacks = 10;
    constexpr int bodyStacks = 6;

    auto pushRing = [&](float y, float radius, Vec3 normal, float vAlong, float region, float bodyT) {
        addRing(vertices, segments, y, radius, normal, vAlong, region, bodyT);
    };

    for (int i = 0; i <= capStacks; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(capStacks);
        const float theta = -kPi * 0.5f + t * (kPi * 0.5f);
        const float radius = std::cos(theta);
        const float y = 0.2f + 0.2f * std::sin(theta);
        pushRing(y, radius, {radius, std::sin(theta), radius}, y, 0.0f, 0.0f);
    }

    for (int i = 1; i <= bodyStacks; ++i) {
        const float bodyT = static_cast<float>(i) / static_cast<float>(bodyStacks);
        const float y = 0.2f + bodyT * 0.6f;
        pushRing(y, 1.0f, {1.0f, 0.0f, 1.0f}, y, 1.0f, bodyT);
    }

    for (int i = 1; i <= capStacks; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(capStacks);
        const float theta = t * (kPi * 0.5f);
        const float radius = std::cos(theta);
        const float y = 0.8f + 0.2f * std::sin(theta);
        pushRing(y, radius, {radius, std::sin(theta), radius}, y, 2.0f, 1.0f);
    }

    const int ringCount = capStacks + bodyStacks + capStacks + 1;
    for (int r = 0; r + 1 < ringCount; ++r) {
        connectRings(indices, segments, r, r + 1);
    }
}

static float gridHalfExtent(int gridCount) {
    return kSpacing * static_cast<float>(std::max(gridCount - 1, 0)) * 0.5f;
}

static void enforceRadiusRange(float& minValue, float& maxValue) {
    minValue = std::clamp(minValue, 0.02f, 1.20f);
    maxValue = std::clamp(maxValue, 0.02f, 1.20f);
    if (maxValue < minValue + 0.01f) {
        maxValue = std::min(1.20f, minValue + 0.01f);
        minValue = std::min(minValue, maxValue - 0.01f);
    }
}

static void rebuildInstances() {
    const int gridCount = gControls.gridCount;
    const float denom = static_cast<float>(std::max(gridCount - 1, 1));
    gHalfGridExtent = gridHalfExtent(gridCount);
    gInstances.clear();
    gInstances.reserve(static_cast<size_t>(gridCount * gridCount));

    for (int z = 0; z < gridCount; ++z) {
        const float zT = static_cast<float>(z) / denom;
        for (int x = 0; x < gridCount; ++x) {
            const float xT = static_cast<float>(x) / denom;
            const float r1 = lerp(gControls.radii.r1Min, gControls.radii.r1Max, xT);
            const float r2 = lerp(gControls.radii.r2Min, gControls.radii.r2Max, zT);
            gInstances.push_back({
                {static_cast<float>(x) * kSpacing - gHalfGridExtent, 0.0f, static_cast<float>(z) * kSpacing - gHalfGridExtent},
                kSphereCenterDistance,
                r1,
                r2,
                {0.35f + 0.35f * xT, 0.48f + 0.28f * zT, 0.78f - 0.24f * xT + 0.08f * zT},
            });
        }
    }
    gInstanceCount = static_cast<uint32_t>(gInstances.size());
}

static void rebuildGridLines() {
    gGridLines.clear();
    const int gridCount = gControls.gridCount;
    const float gridMin = -gHalfGridExtent - kSpacing * 0.35f;
    const float gridMax = gHalfGridExtent + kSpacing * 0.35f;
    gGridLines.reserve(static_cast<size_t>(gridCount * 4));
    for (int i = 0; i < gridCount; ++i) {
        const float p = static_cast<float>(i) * kSpacing - gHalfGridExtent;
        gGridLines.push_back({gridMin, 0.0f, p});
        gGridLines.push_back({gridMax, 0.0f, p});
        gGridLines.push_back({p, 0.0f, gridMin});
        gGridLines.push_back({p, 0.0f, gridMax});
    }
    gGridVertexCount = static_cast<uint32_t>(gGridLines.size());
}

static const char* kCapsuleVertexShader = R"GLSL(#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aVAlong;
layout(location = 3) in float aRegion;
layout(location = 4) in float aBodyT;
layout(location = 5) in vec3 aInstancePosition;
layout(location = 6) in float aHeight;
layout(location = 7) in float aR1;
layout(location = 8) in float aR2;
layout(location = 9) in vec3 aColor;

uniform mat4 uViewProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vColor;

void main() {
    float h = max(aHeight, abs(aR2 - aR1) + 0.001);
    float s = clamp((aR2 - aR1) / h, -0.98, 0.98);
    float q = sqrt(max(1.0 - s * s, 0.0001));
    float tangentAngle = -asin(s);

    float radialLen = length(aPosition.xz);
    vec2 radialDir = radialLen > 0.00001 ? aPosition.xz / radialLen : vec2(1.0, 0.0);

    float localRadius;
    float localY;
    vec3 n;

    if (aRegion < 0.5) {
        float capT = clamp(aPosition.y / 0.2, 0.0, 1.0);
        float theta = -1.57079632679 + (tangentAngle + 1.57079632679) * capT;
        localRadius = aR1 * cos(theta);
        localY = aR1 * sin(theta);
        n = normalize(vec3(radialDir.x * cos(theta), sin(theta), radialDir.y * cos(theta)));
    } else if (aRegion > 1.5) {
        float capT = clamp((aPosition.y - 0.8) / 0.2, 0.0, 1.0);
        float theta = tangentAngle + (1.57079632679 - tangentAngle) * capT;
        localRadius = aR2 * cos(theta);
        localY = h + aR2 * sin(theta);
        n = normalize(vec3(radialDir.x * cos(theta), sin(theta), radialDir.y * cos(theta)));
    } else {
        localRadius = aR1 * q + (aR2 * q - aR1 * q) * aBodyT;
        localY = (-aR1 * s) + ((h - aR2 * s) - (-aR1 * s)) * aBodyT;
        n = normalize(vec3(radialDir.x * q, -s, radialDir.y * q));
    }

    vec3 world = vec3(
        radialDir.x * localRadius + aInstancePosition.x,
        localY + aR1 + aInstancePosition.y,
        radialDir.y * localRadius + aInstancePosition.z
    );

    vWorldPos = world;
    vNormal = n;
    vColor = aColor;
    gl_Position = uViewProj * vec4(world, 1.0);
}
)GLSL";

static const char* kCapsuleFragmentShader = R"GLSL(#version 300 es
precision highp float;

uniform vec3 uCameraPos;

in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vColor;

out vec4 fragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 keyDir = normalize(vec3(-0.45, 0.82, 0.35));
    vec3 fillDir = normalize(vec3(0.75, 0.38, -0.55));
    vec3 topDir = normalize(vec3(0.10, 1.00, 0.15));

    float key = max(dot(N, keyDir), 0.0);
    float fill = max(dot(N, fillDir), 0.0);
    float top = max(dot(N, topDir), 0.0);
    float sky = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0);

    vec3 ambient = mix(vec3(0.13, 0.14, 0.17), vec3(0.24, 0.25, 0.28), sky);
    vec3 diffuse =
        ambient +
        key * vec3(0.52, 0.50, 0.46) +
        fill * vec3(0.13, 0.17, 0.24) +
        top * vec3(0.08, 0.08, 0.07);

    vec3 H = normalize(keyDir + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0) * 0.07;
    vec3 color = vColor * diffuse + vec3(spec) + rim * vec3(0.035, 0.045, 0.060);
    fragColor = vec4(color, 1.0);
}
)GLSL";

static const char* kGridVertexShader = R"GLSL(#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;

uniform mat4 uViewProj;

void main() {
    gl_Position = uViewProj * vec4(aPosition, 1.0);
}
)GLSL";

static const char* kGridFragmentShader = R"GLSL(#version 300 es
precision highp float;

out vec4 fragColor;

void main() {
    fragColor = vec4(0.26, 0.28, 0.30, 1.0);
}
)GLSL";

static GLuint compileShader(GLenum type, const char* source, const char* label) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> log(static_cast<size_t>(std::max(logLength, 1)));
        glGetShaderInfoLog(shader, logLength, nullptr, log.data());
        std::printf("%s shader compile failed: %s\n", label, log.data());
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

static GLuint createProgram(const char* vertexSource, const char* fragmentSource, const char* label) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource, label);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource, label);
    if (!vertexShader || !fragmentShader) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint logLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> log(static_cast<size_t>(std::max(logLength, 1)));
        glGetProgramInfoLog(program, logLength, nullptr, log.data());
        std::printf("%s program link failed: %s\n", label, log.data());
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

static void uploadDynamicSceneBuffers() {
    glBindBuffer(GL_ARRAY_BUFFER, gInstanceBuffer);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(gInstances.size() * sizeof(Instance)),
        gInstances.data(),
        GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, gGridBuffer);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(gGridLines.size() * sizeof(Vec3)),
        gGridLines.data(),
        GL_DYNAMIC_DRAW);
}

static int imguiMouseButtonFromDom(int button) {
    if (button == 2) {
        return ImGuiMouseButton_Right;
    }
    if (button == 1) {
        return ImGuiMouseButton_Middle;
    }
    return ImGuiMouseButton_Left;
}

static EM_BOOL mouseMoveCallback(int, const EmscriptenMouseEvent* event, void*) {
    ImGui::GetIO().AddMousePosEvent(static_cast<float>(event->clientX), static_cast<float>(event->clientY));
    return EM_TRUE;
}

static EM_BOOL mouseButtonCallback(int eventType, const EmscriptenMouseEvent* event, void*) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(static_cast<float>(event->clientX), static_cast<float>(event->clientY));
    io.AddMouseButtonEvent(imguiMouseButtonFromDom(event->button), eventType == EMSCRIPTEN_EVENT_MOUSEDOWN);
    return EM_TRUE;
}

static EM_BOOL wheelCallback(int, const EmscriptenWheelEvent* event, void*) {
    constexpr double kWheelScale = 100.0;
    ImGui::GetIO().AddMouseWheelEvent(
        static_cast<float>(-event->deltaX / kWheelScale),
        static_cast<float>(-event->deltaY / kWheelScale));
    return EM_TRUE;
}

static EM_BOOL touchCallback(int eventType, const EmscriptenTouchEvent* event, void*) {
    ImGuiIO& io = ImGui::GetIO();
    if (event->numTouches > 0) {
        const EmscriptenTouchPoint& touch = event->touches[0];
        io.AddMousePosEvent(static_cast<float>(touch.clientX), static_cast<float>(touch.clientY));
    }

    if (eventType == EMSCRIPTEN_EVENT_TOUCHSTART) {
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    } else if (eventType == EMSCRIPTEN_EVENT_TOUCHEND || eventType == EMSCRIPTEN_EVENT_TOUCHCANCEL) {
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    }
    return EM_TRUE;
}

static EM_BOOL keyCallback(int eventType, const EmscriptenKeyboardEvent* event, void*) {
    if (std::strcmp(event->key, "r") == 0 || std::strcmp(event->key, "R") == 0) {
        gResetPressed = eventType == EMSCRIPTEN_EVENT_KEYDOWN;
    }
    return EM_FALSE;
}

static void installCanvasInputCallbacks() {
    emscripten_set_mousemove_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, false, mouseMoveCallback);
    emscripten_set_mousedown_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, false, mouseButtonCallback);
    emscripten_set_mouseup_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, false, mouseButtonCallback);
    emscripten_set_wheel_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, false, wheelCallback);
    emscripten_set_touchstart_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, false, touchCallback);
    emscripten_set_touchmove_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, false, touchCallback);
    emscripten_set_touchend_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, false, touchCallback);
    emscripten_set_touchcancel_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, false, touchCallback);
    emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, keyCallback);
    emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, keyCallback);
}

static void vertexAttrib(GLuint index, GLint components, GLsizei stride, size_t offset) {
    glEnableVertexAttribArray(index);
    glVertexAttribPointer(index, components, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offset));
}

static void createSceneBuffers() {
    glGenVertexArrays(1, &gCapsuleVao);
    glGenBuffers(1, &gCapsuleVertexBuffer);
    glGenBuffers(1, &gCapsuleIndexBuffer);
    glGenBuffers(1, &gInstanceBuffer);
    glGenVertexArrays(1, &gGridVao);
    glGenBuffers(1, &gGridBuffer);

    glBindVertexArray(gCapsuleVao);

    glBindBuffer(GL_ARRAY_BUFFER, gCapsuleVertexBuffer);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(gVertices.size() * sizeof(Vertex)),
        gVertices.data(),
        GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gCapsuleIndexBuffer);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(gIndices.size() * sizeof(uint32_t)),
        gIndices.data(),
        GL_STATIC_DRAW);

    vertexAttrib(0, 3, sizeof(Vertex), offsetof(Vertex, position));
    vertexAttrib(1, 3, sizeof(Vertex), offsetof(Vertex, normal));
    vertexAttrib(2, 1, sizeof(Vertex), offsetof(Vertex, vAlong));
    vertexAttrib(3, 1, sizeof(Vertex), offsetof(Vertex, region));
    vertexAttrib(4, 1, sizeof(Vertex), offsetof(Vertex, bodyT));

    // Per-instance attributes live in one VBO. The divisor is the key detail:
    // these attributes advance once per capsule, not once per mesh vertex.
    glBindBuffer(GL_ARRAY_BUFFER, gInstanceBuffer);
    vertexAttrib(5, 3, sizeof(Instance), offsetof(Instance, position));
    vertexAttrib(6, 1, sizeof(Instance), offsetof(Instance, height));
    vertexAttrib(7, 1, sizeof(Instance), offsetof(Instance, r1));
    vertexAttrib(8, 1, sizeof(Instance), offsetof(Instance, r2));
    vertexAttrib(9, 3, sizeof(Instance), offsetof(Instance, color));
    for (GLuint attrib = 5; attrib <= 9; ++attrib) {
        glVertexAttribDivisor(attrib, 1);
    }

    glBindVertexArray(gGridVao);
    glBindBuffer(GL_ARRAY_BUFFER, gGridBuffer);
    vertexAttrib(0, 3, sizeof(Vec3), 0);

    glBindVertexArray(0);
    uploadDynamicSceneBuffers();
}

static void updateCameraUniform() {
    const Vec3 eye{
        gCameraTarget.x + gDistance * std::cos(gPitch) * std::sin(gYaw),
        gCameraTarget.y + gDistance * std::sin(gPitch),
        gCameraTarget.z + gDistance * std::cos(gPitch) * std::cos(gYaw),
    };

    const float aspect = static_cast<float>(gWidth) / static_cast<float>(std::max(gHeight, 1u));
    const Mat4 view = lookAt(eye, gCameraTarget, {0.0f, 1.0f, 0.0f});
    const Mat4 proj = perspectiveOpenGL(radians(45.0f), aspect, 0.05f, 1000.0f);
    gViewProj = multiply(proj, view);
    gCameraPos = eye;
}

static void resetCamera() {
    gYaw = radians(38.0f);
    gPitch = radians(28.0f);
    gDistance = std::max(27.0f, gHalfGridExtent * 2.2f);
    gCameraTarget = {0.0f, kSphereCenterDistance * 0.55f, 0.0f};
}

static void updateCameraFromImGui() {
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            const ImVec2 delta = io.MouseDelta;
            gYaw -= delta.x * 0.006f;
            gPitch += delta.y * 0.006f;
            gPitch = std::clamp(gPitch, radians(8.0f), radians(78.0f));
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
            const ImVec2 delta = io.MouseDelta;
            const Vec3 eye{
                gCameraTarget.x + gDistance * std::cos(gPitch) * std::sin(gYaw),
                gCameraTarget.y + gDistance * std::sin(gPitch),
                gCameraTarget.z + gDistance * std::cos(gPitch) * std::cos(gYaw),
            };
            const Vec3 viewDir = normalize(gCameraTarget - eye);
            const Vec3 right = normalize(cross(viewDir, {0.0f, 1.0f, 0.0f}));
            const Vec3 up = normalize(cross(right, viewDir));

            const float displayHeight = std::max(io.DisplaySize.y, 1.0f);
            const float worldPerPixel = 2.0f * gDistance * std::tan(radians(45.0f) * 0.5f) / displayHeight;
            gCameraTarget += right * (-delta.x * worldPerPixel) + up * (delta.y * worldPerPixel);
        }
        if (io.MouseWheel != 0.0f) {
            gDistance *= std::pow(0.88f, io.MouseWheel);
            gDistance = std::clamp(gDistance, 12.0f, 650.0f);
        }
    }
}

static void updateRunningFrameTime() {
    const float frameMs = ImGui::GetIO().DeltaTime * 1000.0f;
    if (frameMs <= 0.0f) {
        return;
    }

    if (gRunningFrameMs <= 0.0f) {
        gRunningFrameMs = frameMs;
    } else {
        gRunningFrameMs += (frameMs - gRunningFrameMs) * 0.05f;
    }
}

static void drawControls() {
    bool instanceChanged = false;
    bool gridChanged = false;

    ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380.0f, 250.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Capsule controls");

    int requestedGridCount = gControls.gridCount;
    gridChanged = ImGui::SliderInt("grid N", &requestedGridCount, 1, 100);
    if (gridChanged) {
        gControls.gridCount = std::clamp(requestedGridCount, 1, 100);
    }
    ImGui::Text("%d x %d = %d capsules", gControls.gridCount, gControls.gridCount, gControls.gridCount * gControls.gridCount);

    instanceChanged |= ImGui::DragFloatRange2(
        "r1 range",
        &gControls.radii.r1Min,
        &gControls.radii.r1Max,
        0.005f,
        0.02f,
        1.20f,
        "min %.3f",
        "max %.3f");
    instanceChanged |= ImGui::DragFloatRange2(
        "r2 range",
        &gControls.radii.r2Min,
        &gControls.radii.r2Max,
        0.005f,
        0.02f,
        1.20f,
        "min %.3f",
        "max %.3f");

    if (ImGui::Button("Reset radii")) {
        gControls.radii = {};
        instanceChanged = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Fit camera")) {
        gDistance = std::max(27.0f, gHalfGridExtent * 2.2f);
    }
    ImGui::SameLine();
    ImGui::Text("h %.2f", kSphereCenterDistance);

    if (gReportedSamples > 1) {
        ImGui::Text("AA: %dx canvas MSAA", gReportedSamples);
    } else {
        ImGui::Text("AA: browser canvas antialias");
    }
    ImGui::Text("running avg frame: %.2f ms (%.1f fps)", gRunningFrameMs, gRunningFrameMs > 0.0f ? 1000.0f / gRunningFrameMs : 0.0f);
    ImGui::End();

    if (instanceChanged || gridChanged) {
        enforceRadiusRange(gControls.radii.r1Min, gControls.radii.r1Max);
        enforceRadiusRange(gControls.radii.r2Min, gControls.radii.r2Max);
        rebuildInstances();
        rebuildGridLines();
        uploadDynamicSceneBuffers();
        if (gridChanged) {
            gDistance = std::max(gDistance, std::max(27.0f, gHalfGridExtent * 2.2f));
        }
    }
}

static void renderScene() {
    glViewport(0, 0, static_cast<GLsizei>(gWidth), static_cast<GLsizei>(gHeight));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    glClearColor(0.08f, 0.085f, 0.095f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(gGridProgram);
    glUniformMatrix4fv(gGridViewProjLocation, 1, GL_FALSE, gViewProj.m);
    glBindVertexArray(gGridVao);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(gGridVertexCount));

    glUseProgram(gCapsuleProgram);
    glUniformMatrix4fv(gCapsuleViewProjLocation, 1, GL_FALSE, gViewProj.m);
    glUniform3f(gCapsuleCameraPosLocation, gCameraPos.x, gCameraPos.y, gCameraPos.z);
    glBindVertexArray(gCapsuleVao);
    glDrawElementsInstanced(
        GL_TRIANGLES,
        static_cast<GLsizei>(gIndexCount),
        GL_UNSIGNED_INT,
        nullptr,
        static_cast<GLsizei>(gInstanceCount));

    glBindVertexArray(0);
}

static void frame() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
    }

    double cssWidth = 0.0;
    double cssHeight = 0.0;
    if (emscripten_get_element_css_size("#canvas", &cssWidth, &cssHeight) == EMSCRIPTEN_RESULT_SUCCESS) {
        int windowWidth = 0;
        int windowHeight = 0;
        SDL_GetWindowSize(gWindow, &windowWidth, &windowHeight);
        const int requestedWidth = std::max(1, static_cast<int>(std::round(cssWidth)));
        const int requestedHeight = std::max(1, static_cast<int>(std::round(cssHeight)));
        if (windowWidth != requestedWidth || windowHeight != requestedHeight) {
            SDL_SetWindowSize(gWindow, requestedWidth, requestedHeight);
        }
    }

    int fbWidth = 0;
    int fbHeight = 0;
    SDL_GL_GetDrawableSize(gWindow, &fbWidth, &fbHeight);
    gWidth = static_cast<uint32_t>(std::max(fbWidth, 1));
    gHeight = static_cast<uint32_t>(std::max(fbHeight, 1));

    if (gResetPressed) {
        resetCamera();
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGuiIO& io = ImGui::GetIO();
    int windowWidth = 0;
    int windowHeight = 0;
    SDL_GetWindowSize(gWindow, &windowWidth, &windowHeight);
    io.DisplaySize = ImVec2(static_cast<float>(std::max(windowWidth, 1)), static_cast<float>(std::max(windowHeight, 1)));
    io.DisplayFramebufferScale = ImVec2(
        static_cast<float>(gWidth) / io.DisplaySize.x,
        static_cast<float>(gHeight) / io.DisplaySize.y);
    ImGui::NewFrame();

    updateCameraFromImGui();
    updateRunningFrameTime();
    drawControls();
    ImGui::Render();
    updateCameraUniform();

    renderScene();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(gWindow);
}

static void setupImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;

    if (!io.Fonts->AddFontFromFileTTF("/fonts/Roboto-Medium.ttf", 16.0f)) {
        io.Fonts->AddFontDefault();
    }

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.FramePadding = ImVec2(8.0f, 5.0f);

    ImGui_ImplOpenGL3_Init("#version 300 es");
    installCanvasInputCallbacks();
}

static void initScene() {
    setupImGui();

    glGetIntegerv(GL_SAMPLES, &gReportedSamples);

    gCapsuleProgram = createProgram(kCapsuleVertexShader, kCapsuleFragmentShader, "capsule");
    gGridProgram = createProgram(kGridVertexShader, kGridFragmentShader, "grid");
    if (!gCapsuleProgram || !gGridProgram) {
        std::printf("Failed to create WebGL2 shader programs\n");
        return;
    }

    gCapsuleViewProjLocation = glGetUniformLocation(gCapsuleProgram, "uViewProj");
    gCapsuleCameraPosLocation = glGetUniformLocation(gCapsuleProgram, "uCameraPos");
    gGridViewProjLocation = glGetUniformLocation(gGridProgram, "uViewProj");

    createSharedUnitCapsuleMesh(gVertices, gIndices);
    gIndexCount = static_cast<uint32_t>(gIndices.size());
    rebuildInstances();
    rebuildGridLines();
    createSceneBuffers();

    resetCamera();
    emscripten_set_main_loop(frame, 0, true);
}

int main() {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0) {
        std::printf("Failed to initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    gWindow = SDL_CreateWindow(
        "WebGL2 instanced tapered capsules",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        static_cast<int>(gWidth),
        static_cast<int>(gHeight),
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!gWindow) {
        std::printf("Failed to create SDL window: %s\n", SDL_GetError());
        return 1;
    }

    gGlContext = SDL_GL_CreateContext(gWindow);
    if (!gGlContext) {
        std::printf("Failed to create WebGL2 context: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_MakeCurrent(gWindow, gGlContext);
    SDL_GL_SetSwapInterval(1);

    const GLubyte* version = glGetString(GL_VERSION);
    const GLubyte* renderer = glGetString(GL_RENDERER);
    std::printf("GL_VERSION: %s\n", version ? reinterpret_cast<const char*>(version) : "unknown");
    std::printf("GL_RENDERER: %s\n", renderer ? reinterpret_cast<const char*>(renderer) : "unknown");

    initScene();
    return 0;
}
