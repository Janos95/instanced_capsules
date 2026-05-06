#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <emscripten.h>
#include <GLFW/glfw3.h>
#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_wgpu.h"

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

struct CameraUniform {
    float viewProj[16];
    float cameraPos[4];
};

static constexpr float kPi = 3.14159265358979323846f;
static constexpr float kSpacing = 2.55f;
static constexpr float kSphereCenterDistance = 2.4f;
static constexpr uint32_t kSceneSampleCount = 4;

static GLFWwindow* gWindow = nullptr;
static wgpu::Instance gInstance;
static wgpu::Adapter gAdapter;
static wgpu::Device gDevice;
static wgpu::Queue gQueue;
static wgpu::Surface gSurface;
static wgpu::TextureFormat gSurfaceFormat = wgpu::TextureFormat::BGRA8Unorm;
static wgpu::TextureView gMsaaColorView;
static wgpu::TextureView gDepthView;
static wgpu::RenderPipeline gCapsulePipeline;
static wgpu::RenderPipeline gGridPipeline;
static wgpu::BindGroupLayout gCameraBindGroupLayout;
static wgpu::BindGroup gCameraBindGroup;
static wgpu::Buffer gCameraBuffer;
static wgpu::Buffer gMeshVertexBuffer;
static wgpu::Buffer gMeshIndexBuffer;
static wgpu::Buffer gInstanceBuffer;
static wgpu::Buffer gGridBuffer;
static uint32_t gWidth = 1280;
static uint32_t gHeight = 900;
static uint32_t gIndexCount = 0;
static uint32_t gInstanceCount = 0;
static uint32_t gGridVertexCount = 0;

static DemoControls gControls;
static float gYaw = 38.0f * kPi / 180.0f;
static float gPitch = 28.0f * kPi / 180.0f;
static float gDistance = 27.0f;
static Vec3 gCameraTarget{0.0f, kSphereCenterDistance * 0.55f, 0.0f};
static float gHalfGridExtent = 0.0f;
static float gRunningFrameMs = 0.0f;

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

static Mat4 perspectiveWebGPU(float fovyRadians, float aspect, float nearPlane, float farPlane) {
    Mat4 r{};
    const float f = 1.0f / std::tan(fovyRadians * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = farPlane / (nearPlane - farPlane);
    r.m[11] = -1.0f;
    r.m[14] = (farPlane * nearPlane) / (nearPlane - farPlane);
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

static wgpu::BufferUsage bufferUsage(std::initializer_list<wgpu::BufferUsage> values) {
    uint64_t bits = 0;
    for (wgpu::BufferUsage value : values) {
        bits |= static_cast<uint64_t>(value);
    }
    return static_cast<wgpu::BufferUsage>(bits);
}

static wgpu::TextureUsage textureUsage(std::initializer_list<wgpu::TextureUsage> values) {
    uint64_t bits = 0;
    for (wgpu::TextureUsage value : values) {
        bits |= static_cast<uint64_t>(value);
    }
    return static_cast<wgpu::TextureUsage>(bits);
}

static wgpu::ShaderStage shaderStage(std::initializer_list<wgpu::ShaderStage> values) {
    uint64_t bits = 0;
    for (wgpu::ShaderStage value : values) {
        bits |= static_cast<uint64_t>(value);
    }
    return static_cast<wgpu::ShaderStage>(bits);
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
    constexpr int segments = 56;
    constexpr int capStacks = 16;
    constexpr int bodyStacks = 12;

    // This is the only capsule/tapered-capsule mesh built on the CPU.
    // It is a reusable reference parameter mesh. The WebGPU vertex shader maps
    // its logical bottom/body/top regions to the convex hull of two spheres using
    // each instance's h, r1, and r2.
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

static wgpu::Buffer createBuffer(const void* data, size_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor desc{};
    desc.size = std::max<size_t>(size, 4);
    desc.usage = usage;
    desc.mappedAtCreation = data != nullptr && size > 0;
    wgpu::Buffer buffer = gDevice.CreateBuffer(&desc);
    if (data && size > 0) {
        std::memcpy(buffer.GetMappedRange(0, size), data, size);
        buffer.Unmap();
    }
    return buffer;
}

static void uploadDynamicSceneBuffers() {
    gInstanceBuffer = createBuffer(
        gInstances.data(),
        gInstances.size() * sizeof(Instance),
        bufferUsage({wgpu::BufferUsage::Vertex, wgpu::BufferUsage::CopyDst}));
    gGridBuffer = createBuffer(
        gGridLines.data(),
        gGridLines.size() * sizeof(Vec3),
        bufferUsage({wgpu::BufferUsage::Vertex, wgpu::BufferUsage::CopyDst}));
}

static const char* kCapsuleWGSL = R"WGSL(
struct Camera {
    viewProj: mat4x4<f32>,
    cameraPos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) vAlong: f32,
    @location(3) region: f32,
    @location(4) bodyT: f32,
    @location(5) instancePosition: vec3<f32>,
    @location(6) height: f32,
    @location(7) r1: f32,
    @location(8) r2: f32,
    @location(9) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn capsule_vs(input: VertexInput) -> VertexOutput {
    let h = max(input.height, abs(input.r2 - input.r1) + 0.001);
    let s = clamp((input.r2 - input.r1) / h, -0.98, 0.98);
    let q = sqrt(max(1.0 - s * s, 0.0001));
    let tangentAngle = -asin(s);

    let radialLen = length(input.position.xz);
    let radialDir = select(vec2<f32>(1.0, 0.0), input.position.xz / radialLen, radialLen > 0.00001);

    var localRadius: f32;
    var localY: f32;
    var n: vec3<f32>;

    if (input.region < 0.5) {
        let capT = clamp(input.position.y / 0.2, 0.0, 1.0);
        let theta = -1.57079632679 + (tangentAngle + 1.57079632679) * capT;
        localRadius = input.r1 * cos(theta);
        localY = input.r1 * sin(theta);
        n = normalize(vec3<f32>(radialDir.x * cos(theta), sin(theta), radialDir.y * cos(theta)));
    } else if (input.region > 1.5) {
        let capT = clamp((input.position.y - 0.8) / 0.2, 0.0, 1.0);
        let theta = tangentAngle + (1.57079632679 - tangentAngle) * capT;
        localRadius = input.r2 * cos(theta);
        localY = h + input.r2 * sin(theta);
        n = normalize(vec3<f32>(radialDir.x * cos(theta), sin(theta), radialDir.y * cos(theta)));
    } else {
        localRadius = input.r1 * q + (input.r2 * q - input.r1 * q) * input.bodyT;
        localY = (-input.r1 * s) + ((h - input.r2 * s) - (-input.r1 * s)) * input.bodyT;
        n = normalize(vec3<f32>(radialDir.x * q, -s, radialDir.y * q));
    }

    let world = vec3<f32>(
        radialDir.x * localRadius + input.instancePosition.x,
        localY + input.r1 + input.instancePosition.y,
        radialDir.y * localRadius + input.instancePosition.z
    );

    var output: VertexOutput;
    output.clipPosition = camera.viewProj * vec4<f32>(world, 1.0);
    output.worldPos = world;
    output.normal = n;
    output.color = input.color;
    return output;
}

@fragment
fn capsule_fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(input.normal);
    let V = normalize(camera.cameraPos.xyz - input.worldPos);
    let keyDir = normalize(vec3<f32>(-0.45, 0.82, 0.35));
    let fillDir = normalize(vec3<f32>(0.75, 0.38, -0.55));
    let topDir = normalize(vec3<f32>(0.10, 1.00, 0.15));

    let key = max(dot(N, keyDir), 0.0);
    let fill = max(dot(N, fillDir), 0.0);
    let top = max(dot(N, topDir), 0.0);
    let sky = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    let rim = pow(1.0 - max(dot(N, V), 0.0), 2.0);

    let ambient = vec3<f32>(0.13, 0.14, 0.17) + (vec3<f32>(0.24, 0.25, 0.28) - vec3<f32>(0.13, 0.14, 0.17)) * sky;
    let diffuse =
        ambient +
        key * vec3<f32>(0.52, 0.50, 0.46) +
        fill * vec3<f32>(0.13, 0.17, 0.24) +
        top * vec3<f32>(0.08, 0.08, 0.07);

    let H = normalize(keyDir + V);
    let spec = pow(max(dot(N, H), 0.0), 64.0) * 0.07;
    let color = input.color * diffuse + vec3<f32>(spec) + rim * vec3<f32>(0.035, 0.045, 0.060);
    return vec4<f32>(color, 1.0);
}
)WGSL";

static const char* kGridWGSL = R"WGSL(
struct Camera {
    viewProj: mat4x4<f32>,
    cameraPos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
};

@vertex
fn grid_vs(@location(0) position: vec3<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.clipPosition = camera.viewProj * vec4<f32>(position, 1.0);
    return output;
}

@fragment
fn grid_fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.26, 0.28, 0.30, 1.0);
}
)WGSL";

static wgpu::ShaderModule createShader(const char* wgslSource) {
    wgpu::ShaderSourceWGSL source{};
    source.code = wgslSource;
    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &source;
    return gDevice.CreateShaderModule(&desc);
}

static void createCameraResources() {
    wgpu::BindGroupLayoutEntry entry{};
    entry.binding = 0;
    entry.visibility = shaderStage({wgpu::ShaderStage::Vertex, wgpu::ShaderStage::Fragment});
    entry.buffer.type = wgpu::BufferBindingType::Uniform;
    entry.buffer.minBindingSize = sizeof(CameraUniform);

    wgpu::BindGroupLayoutDescriptor layoutDesc{};
    layoutDesc.entryCount = 1;
    layoutDesc.entries = &entry;
    gCameraBindGroupLayout = gDevice.CreateBindGroupLayout(&layoutDesc);

    gCameraBuffer = createBuffer(
        nullptr,
        sizeof(CameraUniform),
        bufferUsage({wgpu::BufferUsage::Uniform, wgpu::BufferUsage::CopyDst}));

    wgpu::BindGroupEntry bindEntry{};
    bindEntry.binding = 0;
    bindEntry.buffer = gCameraBuffer;
    bindEntry.offset = 0;
    bindEntry.size = sizeof(CameraUniform);

    wgpu::BindGroupDescriptor bindDesc{};
    bindDesc.layout = gCameraBindGroupLayout;
    bindDesc.entryCount = 1;
    bindDesc.entries = &bindEntry;
    gCameraBindGroup = gDevice.CreateBindGroup(&bindDesc);
}

static wgpu::RenderPipeline createCapsulePipeline() {
    wgpu::ShaderModule shader = createShader(kCapsuleWGSL);

    std::array<wgpu::VertexAttribute, 5> meshAttrs{};
    meshAttrs[0] = {nullptr, wgpu::VertexFormat::Float32x3, offsetof(Vertex, position), 0};
    meshAttrs[1] = {nullptr, wgpu::VertexFormat::Float32x3, offsetof(Vertex, normal), 1};
    meshAttrs[2] = {nullptr, wgpu::VertexFormat::Float32, offsetof(Vertex, vAlong), 2};
    meshAttrs[3] = {nullptr, wgpu::VertexFormat::Float32, offsetof(Vertex, region), 3};
    meshAttrs[4] = {nullptr, wgpu::VertexFormat::Float32, offsetof(Vertex, bodyT), 4};

    std::array<wgpu::VertexAttribute, 5> instanceAttrs{};
    instanceAttrs[0] = {nullptr, wgpu::VertexFormat::Float32x3, offsetof(Instance, position), 5};
    instanceAttrs[1] = {nullptr, wgpu::VertexFormat::Float32, offsetof(Instance, height), 6};
    instanceAttrs[2] = {nullptr, wgpu::VertexFormat::Float32, offsetof(Instance, r1), 7};
    instanceAttrs[3] = {nullptr, wgpu::VertexFormat::Float32, offsetof(Instance, r2), 8};
    instanceAttrs[4] = {nullptr, wgpu::VertexFormat::Float32x3, offsetof(Instance, color), 9};

    std::array<wgpu::VertexBufferLayout, 2> layouts{};
    layouts[0].arrayStride = sizeof(Vertex);
    layouts[0].stepMode = wgpu::VertexStepMode::Vertex;
    layouts[0].attributeCount = meshAttrs.size();
    layouts[0].attributes = meshAttrs.data();
    layouts[1].arrayStride = sizeof(Instance);
    layouts[1].stepMode = wgpu::VertexStepMode::Instance;
    layouts[1].attributeCount = instanceAttrs.size();
    layouts[1].attributes = instanceAttrs.data();

    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &gCameraBindGroupLayout;
    wgpu::PipelineLayout pipelineLayout = gDevice.CreatePipelineLayout(&pipelineLayoutDesc);

    wgpu::ColorTargetState colorTarget{};
    colorTarget.format = gSurfaceFormat;

    wgpu::FragmentState fragment{};
    fragment.module = shader;
    fragment.entryPoint = "capsule_fs";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;

    wgpu::DepthStencilState depth{};
    depth.format = wgpu::TextureFormat::Depth32Float;
    depth.depthWriteEnabled = true;
    depth.depthCompare = wgpu::CompareFunction::Less;

    wgpu::RenderPipelineDescriptor desc{};
    desc.layout = pipelineLayout;
    desc.vertex.module = shader;
    desc.vertex.entryPoint = "capsule_vs";
    desc.vertex.bufferCount = layouts.size();
    desc.vertex.buffers = layouts.data();
    desc.fragment = &fragment;
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.frontFace = wgpu::FrontFace::CCW;
    desc.primitive.cullMode = wgpu::CullMode::Back;
    desc.multisample.count = kSceneSampleCount;
    desc.depthStencil = &depth;
    return gDevice.CreateRenderPipeline(&desc);
}

static wgpu::RenderPipeline createGridPipeline() {
    wgpu::ShaderModule shader = createShader(kGridWGSL);

    wgpu::VertexAttribute positionAttr{};
    positionAttr.format = wgpu::VertexFormat::Float32x3;
    positionAttr.offset = 0;
    positionAttr.shaderLocation = 0;

    wgpu::VertexBufferLayout layout{};
    layout.arrayStride = sizeof(Vec3);
    layout.stepMode = wgpu::VertexStepMode::Vertex;
    layout.attributeCount = 1;
    layout.attributes = &positionAttr;

    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &gCameraBindGroupLayout;
    wgpu::PipelineLayout pipelineLayout = gDevice.CreatePipelineLayout(&pipelineLayoutDesc);

    wgpu::ColorTargetState colorTarget{};
    colorTarget.format = gSurfaceFormat;

    wgpu::FragmentState fragment{};
    fragment.module = shader;
    fragment.entryPoint = "grid_fs";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;

    wgpu::DepthStencilState depth{};
    depth.format = wgpu::TextureFormat::Depth32Float;
    depth.depthWriteEnabled = false;
    depth.depthCompare = wgpu::CompareFunction::LessEqual;

    wgpu::RenderPipelineDescriptor desc{};
    desc.layout = pipelineLayout;
    desc.vertex.module = shader;
    desc.vertex.entryPoint = "grid_vs";
    desc.vertex.bufferCount = 1;
    desc.vertex.buffers = &layout;
    desc.fragment = &fragment;
    desc.primitive.topology = wgpu::PrimitiveTopology::LineList;
    desc.primitive.frontFace = wgpu::FrontFace::CCW;
    desc.primitive.cullMode = wgpu::CullMode::None;
    desc.multisample.count = kSceneSampleCount;
    desc.depthStencil = &depth;
    return gDevice.CreateRenderPipeline(&desc);
}

static void configureSurface(uint32_t width, uint32_t height) {
    gWidth = std::max(width, 1u);
    gHeight = std::max(height, 1u);

    wgpu::SurfaceConfiguration config{};
    config.device = gDevice;
    config.format = gSurfaceFormat;
    config.usage = wgpu::TextureUsage::RenderAttachment;
    config.width = gWidth;
    config.height = gHeight;
    config.alphaMode = wgpu::CompositeAlphaMode::Auto;
    config.presentMode = wgpu::PresentMode::Fifo;
    gSurface.Configure(&config);

    wgpu::TextureDescriptor msaaColorDesc{};
    msaaColorDesc.usage = wgpu::TextureUsage::RenderAttachment;
    msaaColorDesc.size = {gWidth, gHeight, 1};
    msaaColorDesc.format = gSurfaceFormat;
    msaaColorDesc.sampleCount = kSceneSampleCount;
    gMsaaColorView = gDevice.CreateTexture(&msaaColorDesc).CreateView();

    wgpu::TextureDescriptor depthDesc{};
    depthDesc.usage = wgpu::TextureUsage::RenderAttachment;
    depthDesc.size = {gWidth, gHeight, 1};
    depthDesc.format = wgpu::TextureFormat::Depth32Float;
    depthDesc.sampleCount = kSceneSampleCount;
    gDepthView = gDevice.CreateTexture(&depthDesc).CreateView();
}

static void updateCameraUniform() {
    const Vec3 eye{
        gCameraTarget.x + gDistance * std::cos(gPitch) * std::sin(gYaw),
        gCameraTarget.y + gDistance * std::sin(gPitch),
        gCameraTarget.z + gDistance * std::cos(gPitch) * std::cos(gYaw),
    };

    const float aspect = static_cast<float>(gWidth) / static_cast<float>(std::max(gHeight, 1u));
    const Mat4 view = lookAt(eye, gCameraTarget, {0.0f, 1.0f, 0.0f});
    const Mat4 proj = perspectiveWebGPU(radians(45.0f), aspect, 0.05f, 1000.0f);
    const Mat4 viewProj = multiply(proj, view);

    CameraUniform camera{};
    std::memcpy(camera.viewProj, viewProj.m, sizeof(viewProj.m));
    camera.cameraPos[0] = eye.x;
    camera.cameraPos[1] = eye.y;
    camera.cameraPos[2] = eye.z;
    camera.cameraPos[3] = 1.0f;
    gQueue.WriteBuffer(gCameraBuffer, 0, &camera, sizeof(camera));
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

            // Move the look-at target so the world follows the right-drag motion.
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

    ImGui::Text("capsules: one WebGPU DrawIndexed(%u, %u)", gIndexCount, gInstanceCount);
    ImGui::Text("AA: %ux MSAA", kSceneSampleCount);
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

static void frame() {
    glfwPollEvents();

    int fbWidth = 0;
    int fbHeight = 0;
    glfwGetFramebufferSize(gWindow, &fbWidth, &fbHeight);
    if (fbWidth > 0 && fbHeight > 0 && (static_cast<uint32_t>(fbWidth) != gWidth || static_cast<uint32_t>(fbHeight) != gHeight)) {
        ImGui_ImplWGPU_InvalidateDeviceObjects();
        configureSurface(static_cast<uint32_t>(fbWidth), static_cast<uint32_t>(fbHeight));
        ImGui_ImplWGPU_CreateDeviceObjects();
    }

    if (glfwGetKey(gWindow, GLFW_KEY_R) == GLFW_PRESS) {
        resetCamera();
    }

    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    updateCameraFromImGui();
    updateRunningFrameTime();
    drawControls();
    ImGui::Render();
    updateCameraUniform();

    wgpu::SurfaceTexture surfaceTexture{};
    gSurface.GetCurrentTexture(&surfaceTexture);
    if (!surfaceTexture.texture) {
        return;
    }

    wgpu::TextureView backbuffer = surfaceTexture.texture.CreateView();

    wgpu::RenderPassColorAttachment sceneColorAttachment{};
    sceneColorAttachment.view = gMsaaColorView;
    sceneColorAttachment.resolveTarget = backbuffer;
    sceneColorAttachment.loadOp = wgpu::LoadOp::Clear;
    sceneColorAttachment.storeOp = wgpu::StoreOp::Discard;
    sceneColorAttachment.clearValue = {0.08, 0.085, 0.095, 1.0};

    wgpu::RenderPassDepthStencilAttachment depthAttachment{};
    depthAttachment.view = gDepthView;
    depthAttachment.depthLoadOp = wgpu::LoadOp::Clear;
    depthAttachment.depthStoreOp = wgpu::StoreOp::Discard;
    depthAttachment.depthClearValue = 1.0f;

    wgpu::RenderPassDescriptor scenePassDesc{};
    scenePassDesc.colorAttachmentCount = 1;
    scenePassDesc.colorAttachments = &sceneColorAttachment;
    scenePassDesc.depthStencilAttachment = &depthAttachment;

    wgpu::CommandEncoder encoder = gDevice.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&scenePassDesc);

    pass.SetPipeline(gGridPipeline);
    pass.SetBindGroup(0, gCameraBindGroup);
    pass.SetVertexBuffer(0, gGridBuffer);
    pass.Draw(gGridVertexCount);

    pass.SetPipeline(gCapsulePipeline);
    pass.SetBindGroup(0, gCameraBindGroup);
    pass.SetVertexBuffer(0, gMeshVertexBuffer);
    pass.SetVertexBuffer(1, gInstanceBuffer);
    pass.SetIndexBuffer(gMeshIndexBuffer, wgpu::IndexFormat::Uint32);
    pass.DrawIndexed(gIndexCount, gInstanceCount);

    pass.End();

    wgpu::RenderPassColorAttachment uiColorAttachment{};
    uiColorAttachment.view = backbuffer;
    uiColorAttachment.loadOp = wgpu::LoadOp::Load;
    uiColorAttachment.storeOp = wgpu::StoreOp::Store;

    wgpu::RenderPassDescriptor uiPassDesc{};
    uiPassDesc.colorAttachmentCount = 1;
    uiPassDesc.colorAttachments = &uiColorAttachment;

    wgpu::RenderPassEncoder uiPass = encoder.BeginRenderPass(&uiPassDesc);
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), uiPass.Get());
    uiPass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    gQueue.Submit(1, &commands);
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

    ImGui_ImplGlfw_InitForOther(gWindow, true);
    ImGui_ImplGlfw_InstallEmscriptenCallbacks(gWindow, "#canvas");

    ImGui_ImplWGPU_InitInfo initInfo{};
    initInfo.Device = gDevice.Get();
    initInfo.NumFramesInFlight = 3;
    initInfo.RenderTargetFormat = static_cast<WGPUTextureFormat>(gSurfaceFormat);
    initInfo.DepthStencilFormat = WGPUTextureFormat_Undefined;
    ImGui_ImplWGPU_Init(&initInfo);
}

static void initScene() {
    gQueue = gDevice.GetQueue();

    wgpu::SurfaceCapabilities capabilities{};
    gSurface.GetCapabilities(gAdapter, &capabilities);
    if (capabilities.formatCount > 0) {
        gSurfaceFormat = capabilities.formats[0];
    }

    int fbWidth = 0;
    int fbHeight = 0;
    glfwGetFramebufferSize(gWindow, &fbWidth, &fbHeight);
    configureSurface(static_cast<uint32_t>(std::max(fbWidth, 1)), static_cast<uint32_t>(std::max(fbHeight, 1)));

    setupImGui();

    createSharedUnitCapsuleMesh(gVertices, gIndices);
    gIndexCount = static_cast<uint32_t>(gIndices.size());
    rebuildInstances();
    rebuildGridLines();

    gMeshVertexBuffer = createBuffer(
        gVertices.data(),
        gVertices.size() * sizeof(Vertex),
        bufferUsage({wgpu::BufferUsage::Vertex, wgpu::BufferUsage::CopyDst}));
    gMeshIndexBuffer = createBuffer(
        gIndices.data(),
        gIndices.size() * sizeof(uint32_t),
        bufferUsage({wgpu::BufferUsage::Index, wgpu::BufferUsage::CopyDst}));
    uploadDynamicSceneBuffers();

    createCameraResources();
    gCapsulePipeline = createCapsulePipeline();
    gGridPipeline = createGridPipeline();

    resetCamera();
    emscripten_set_main_loop(frame, 0, true);
}

static void requestDeviceAndStart() {
    gInstance.RequestAdapter(nullptr, wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message) {
            if (message.length) {
                std::printf("RequestAdapter: %.*s\n", static_cast<int>(message.length), message.data);
            }
            if (status != wgpu::RequestAdapterStatus::Success) {
                std::printf("WebGPU adapter request failed\n");
                return;
            }

            gAdapter = adapter;
            wgpu::DeviceDescriptor desc{};
            desc.SetUncapturedErrorCallback(
                [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView msg) {
                    std::printf("WebGPU error type=%d: %.*s\n", static_cast<int>(type), static_cast<int>(msg.length), msg.data);
                });

            gAdapter.RequestDevice(&desc, wgpu::CallbackMode::AllowSpontaneous,
                [](wgpu::RequestDeviceStatus deviceStatus, wgpu::Device device, wgpu::StringView deviceMessage) {
                    if (deviceMessage.length) {
                        std::printf("RequestDevice: %.*s\n", static_cast<int>(deviceMessage.length), deviceMessage.data);
                    }
                    if (deviceStatus != wgpu::RequestDeviceStatus::Success) {
                        std::printf("WebGPU device request failed\n");
                        return;
                    }

                    gDevice = device;
                    initScene();
                });
        });
}

int main() {
    if (!glfwInit()) {
        std::printf("Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    gWindow = glfwCreateWindow(static_cast<int>(gWidth), static_cast<int>(gHeight), "WebGPU instanced tapered capsules", nullptr, nullptr);
    if (!gWindow) {
        std::printf("Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }

    gInstance = wgpu::Instance(wgpuCreateInstance(nullptr));

    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvasDesc{};
    canvasDesc.selector = "#canvas";
    wgpu::SurfaceDescriptor surfaceDesc{};
    surfaceDesc.nextInChain = &canvasDesc;
    gSurface = gInstance.CreateSurface(&surfaceDesc);

    requestDeviceAndStart();
    return 0;
}
