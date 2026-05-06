# WebGPU Instanced Tapered Capsules

This demo renders a grid of tapered capsules in the browser with WebGPU and Dear ImGui.

Live page: [https://janosmeny.com/instanced_capsules/](https://janosmeny.com/instanced_capsules/)

The renderer uses:

- one reusable reference capsule mesh
- one instance buffer containing `position`, sphere-center distance `height`/`h`, `r1`, `r2`, and `color`
- one indexed instanced WebGPU draw call for all capsules:
- 4x MSAA for the scene pass, resolved before the ImGui overlay pass

```cpp
pass.DrawIndexed(indexCount, instanceCount);
```

Each capsule is the convex hull of two Y-axis-aligned spheres: bottom sphere radius `r1`, top sphere radius `r2`, with centers `h` units apart. The CPU does not generate per-capsule meshes; the WGSL vertex shader deforms the shared reference mesh using per-instance attributes.

## Build

```bash
source /Users/janos/Projects/emsdk/emsdk_env.sh
emcmake cmake -S . -B build-web
cmake --build build-web
```

Run locally:

```bash
python3 -m http.server 8000 -d build-web/web
```

Open [http://localhost:8000](http://localhost:8000).

## Controls

- Drag left mouse button: orbit camera
- Drag right mouse button: pan camera
- Mouse wheel: zoom
- ImGui panel: edit grid `N`, min/max `r1`, min/max `r2`, and view running average frame time
- `R`: reset camera
