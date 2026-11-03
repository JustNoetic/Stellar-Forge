# Stellar Forge

**Stellar Forge** is a high-precision, real-time 3D astrophysical simulator, NASA SPICE ephemeris visualizer, and physically based planetary rendering engine written in **C++20** and **Vulkan 1.3**.

Designed from the ground up around modern GPU-driven pipelines and double-precision celestial mechanics, Stellar Forge seamlessly bridges scales from interplanetary orbital dynamics ($>10^2\text{ AU}$) down to meter-scale oblate planetary terrain, spectral ocean waves, and radiative-transfer atmospheres.

---

## Key Features

### 1. Modern Vulkan 1.3 Rendering Engine
- **Zero-Legacy Vulkan 1.3 Core**: Built on `volk` and `VulkanMemoryAllocator` (VMA) using Dynamic Rendering (`vkCmdBeginRendering`), Synchronization2 (`vkCmdPipelineBarrier2`), Push Descriptors (`vkCmdPushDescriptorSetKHR`), and Buffer Device Address—with zero `VkRenderPass` or `VkFramebuffer` overhead.
- **Reverse-Z & Far-Field Hybrid Precision**: Infinite far-plane Reverse-Z depth (`[1.0 -> 0.0]`, `VK_COMPARE_OP_GREATER_OR_EQUAL`) paired with FP64 camera-relative CPU transforms and body-relative GPU vertex/ray offsets to eliminate floating-point jitter across astronomical distances.
- **Runtime Shader Preprocessor & SPIR-V Cache**: Automatic `#include` dependency tracking and on-disk SPIR-V caching (`ShaderCache`) via `glslang`.
- **Order-Independent Transparency (OIT)**: Per-pixel linked-list A-buffer (16-byte records with early Reverse-Z culling) resolving multi-layer planetary rings, cloud shells, and atmospheric transmittance ($T \cdot P + \alpha \cdot S$).
- **HDR Post-Processing & Diffraction Bloom**: ACES filmic tone mapping fused with a 5-level multiscale bloom halo and sparse, flux-conserving 4-spike diffraction sprites (`BloomRenderer`) driven by tile luminance reduction.

### 2. Precision Orbital Mechanics & NASA SPICE Ephemeris
- **IAS15 15th-Order Adaptive Integrator**: High-accuracy Gauss-Radau N-body numerical integration featuring 1st-Order Post-Newtonian (1PN) General Relativity corrections and $J_2 / J_4$ zonal harmonic oblateness perturbations.
- **NASA NAIF CSPICE Playback & Export**:
  - Live trajectory evaluation from standard NASA `.bsp`, `.tls`, and `.tpc` kernels (`SpiceEphemeris`), including spacecraft missions (*Artemis II Orion*, *Cassini*, *JWST*).
  - Built-in **DAF/SPK Segment Inspector** for previewing custom `.bsp` coverage intervals and bodies, plus Type 2 / Type 13 SPK trajectory export (`SpkExporter`).
  - Integrated multi-threaded **WinHTTP Kernel Downloader** (`SpiceKernelManager`) that fetches missing NAIF kernels on demand with live progress and bandwidth telemetry.
- **Jacobi Hierarchy & Dynamic Reference Frames**: Analytical $J_2$ and third-body nodal/apsidal precession in Keplerian mode, paired with a 15 Hz Hill-sphere AABB reference hierarchy tracker (`ReferenceHierarchyTracker`) with 90%/110% entry/exit hysteresis.
- **Asynchronous Simulation Worker**: Lock-free triple-buffered snapshot pipeline (`SimulationRunner`) decoupling high-frequency physics integration from rendering.

### 3. Physically Based Atmospheres, Eclipse Optics & Shadows
- **Bruneton / Hillaire Radiative Transfer**: Precomputed transmittance (`256×64`), multiple scattering (`32×32`), sky-view (`192×108`), and 3D aerial perspective (`192×108×64`) LUTs with schema v2 Mie/aerosol particle layers and Karkoschka (1994) methane absorption spectra.
- **Volumetric Atmospheric Shadows & Deficit VRS**:
  - Evaluates solar loss fraction deficits across planetary rings and eclipsing moons with software **Variable Rate Shading (VRS)** (`2×`–`16×` coarse grids with two-layer transparent prefix reconstruction).
  - Supports both **deterministic analytic strata** (`ring_swept_filter` for crisp single-frame ring shadows) and **64-frame progressive spatiotemporal blue-noise accumulation** with guarded rotation-only sky history reuse.
- **Finite Stellar Disks**: Physical angular stellar radius integration with limb darkening, visible-disk horizon transmission, and small-source LUT specialization.
- **Atmospheric Refraction & Eclipse Illumination**:
  - Per-frame observer refraction table (`observer_table.comp`) bending surface, cloud, and point-source rays across the horizon.
  - Background-baked `256×64` log-RGB **Eclipse Refraction LUTs** (`EclipseRefractionRenderer`) modeling refracted twilight/umbral fill lighting ("blood moon" illumination) on eclipsed surfaces and rings without per-pixel ray marching.

### 4. Planetary Rings & Secondary Lighting (Planetshine / Ringshine)
- **Multi-Layer Ring Optics**: 3-parameter Double Henyey-Greenstein (DHG) phase functions, slab multiple scattering, 1D procedural band synthesis (`generate_procedural_ring_profile`), and 2D prefix shadow LUTs.
- **Ringshine & Planetshine**:
  - Single-pass `65×65` half-longitude `RGBA32F` **Ringshine** irradiance and scalar fluence bakes per `(host × star)` pair, illuminating both planetary surfaces and atmospheres.
  - Material-aware **Planetshine** supporting up to 4 secondary reflectors per receiver with `128×64` baked lat-long irradiance maps for Lambertian bodies and horizon-culled per-fragment evaluation for Hapke regolith surfaces.

### 5. Oblate Cube-Sphere Terrain, FFT Oceans & Clouds
- **Oblate Quadtree Terrain LOD**: 6-face oblate cube-sphere quadtree (`Quadtree`) with continuous spatial morphing, a 64 MiB GPU patch geometry cache (`GeometryCache`), and async streaming of `128×128` (+2-texel gutter) cubic B-spline elevation DEMs, lossless sRGB albedo tiles, and `RGBA16_SNORM` normal maps.
- **Spectral FFT Ocean Waves**: Five `256×256` JONSWAP FFT cascades (`2 m` to `77 km` wavelengths) for the focused body below `1,000 km` altitude, featuring parallel-transported tangent frames, prefiltered slope/Jacobian moment mips, Cox-Munk residual roughness, Jacobian-quantile whitecap foam, and Bruneton Fresnel reflectance.
- **Cumulus Cloud Shells & Volumetric Layers**: Supports fast 2D oblate texture shells and ray-marched volumetric cumulus layers driven by up to `16384×8192` 14-mip `R8` coverage pyramids (`CloudMaps`), with lifting condensation level (LCL) automatic base altitude calculation.

### 6. Astrophysics, Gaia DR3 Starfield & Comparative Tools
- **Gaia DR3 Star Catalog (~475k Stars)**: GPU compute frustum culling (`starfield_cull.comp`) and `vkCmdDrawIndirect` rendering with flux-normalized 3 px point-spread functions and smooth subpixel mesh-to-point transitions.
- **Astrophysical Star Creator**: Interactive stellar evolution and thermodynamics solver (`StarCalculator`) computing main-sequence tracks, Stefan-Boltzmann luminosity, Roche rotational oblateness limits, and Planckian blackbody chromaticity.
- **Object Size Comparator**: Side-by-side physical scale lineup across any number of saved or live systems with independent orthographic pan/zoom, parallel-ray atmospheres, limb-darkened stars, and filtered rings.
- **Live System Comparator**: Simultaneous multi-system simulation (`ComparisonSet`) with independent `SimulationRunner` threads, spatial offset or orbital overlay modes, per-system starlight isolation switches, and a live orbital/equilibrium-temperature comparison table.

---

## Controls & Shortcuts

| Input | Action |
| :--- | :--- |
| **Right Mouse Drag** | Orbit camera around tracked target / look around |
| **Scroll Wheel** | Zoom in / out (`Shift + Scroll` for fine zoom / flight speed adjustment) |
| **`W` `A` `S` `D` / `Q` `E`** | Free-flight camera translation & roll |
| **`F`** | Snap camera focus to currently tracked body (or fit view in Size Comparator) |
| **`Space`** | Pause / resume simulation clock |
| **`Up` / `Right` Arrow** | Double time warp factor ($2\times$) |
| **`Down` / `Left` Arrow** | Halve time warp factor ($0.5\times$) |
| **`R`** | Reset time warp to $1\times$ real-time |
| **`Backspace`** | Reverse time direction |
| **`=` / `-`** | Increase / decrease HDR camera exposure (`Shift` for $2\times$ steps) |
| **`H`** | Toggle ImGui workspace UI visibility |
| **`F11`** | Toggle borderless fullscreen on current monitor |
| **`Esc`** | Clear body selection / close active modal |
| **Drag & Drop Image** | Stage dropped albedo or DEM heightmap image directly into the Body Inspector's Surface Baker |

---

## Building from Source

### Prerequisites
- **OS**: Windows 10 / 11 (64-bit)
- **Compiler**: C++20 compiler (MSVC 2022+ or MinGW-w64 GCC 13+ / Clang 17+)
- **Build System**: [CMake 3.28+](https://cmake.org/) and [Ninja](https://ninja-build.org/)
- **GPU**: Vulkan 1.3 capable GPU and driver
- **Python 3.10+** *(Optional)*: Required only for offline texture/DEM tile baking (`numpy`, `Pillow`, and optionally `cupy-cuda12x` for GPU-accelerated baking)

All C++ third-party dependencies (`Vulkan-Headers`, `volk`, `VMA`, `GLFW 3.4`, `GLM`, `nlohmann/json`, `Dear ImGui (docking)`, and `glslang`) are automatically fetched at configure time via CMake `FetchContent`.

### Quick Build & Run
Run the included batch script to configure, build in parallel with Ninja, and launch:
```bat
.\run.bat
```

Or build manually via CMake:
```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
.\build\bin\StellarForge.exe
```

### CMake Build Options
| Option | Default | Description |
| :--- | :---: | :--- |
| `STELLAR_BUILD_TESTS` | `ON` | Build CPU numerical, state, orbit, and hierarchy contract tests (`build/bin/tests/`) |
| `STELLAR_BUILD_RENDER_TESTS` | `OFF` | Build GPU rendering contract and visual regression suites |
| `STELLAR_TEST_EPHEMERIS` | `OFF` | Enable SPICE ephemeris tests against locally installed NAIF kernels |

### Creating a Self-Contained Distribution (`dist/`)
```powershell
cmake --install build --prefix dist
```

---

## External Assets: NAIF SPICE Kernels & Planetary Texture Tiles

To keep the Git repository lightweight and fast to clone, multi-gigabyte binary datasets (**~2.1 GB** of NASA NAIF SPICE kernels and **~6.9 GB** of baked planetary surface/elevation tile pyramids) are excluded from version control. Stellar Forge degrades gracefully when optional packs are absent and provides built-in workflows to acquire or generate them:

### 1. NASA NAIF SPICE Kernels (`data/kernels/`)
- **Included in Repo**: The CSPICE shared runtime (`data/kernels/cspice.dll`), license, and compact mission kernels (`artemis2.bsp`).
- **Auto-Downloaded on Demand**: When you switch to **Ephemeris Mode** in the UI, `SpiceKernelManager` automatically checks for missing default kernels (`de440s.bsp`, `jup365.bsp`, `sat441.bsp`, `mar099s.bsp`, `nep095.bsp`, `plu060.bsp`, `naif0012.tls`, `pck00010.tpc`) and downloads them directly from NASA's NAIF servers (`https://naif.jpl.nasa.gov/pub/naif/generic_kernels/`) into `data/kernels/default/`.
- **Custom `.bsp` Files**: Drop any NASA/ESA `.bsp` kernel into `data/kernels/additional/` or open it via **Import Ephemeris Kernel (.bsp)** in the UI.

### 2. Planetary Surface, Elevation & Cloud Tile Packs (`data/tiles/` & `textures/`)
- **Graceful Fallback**: Without baked tiles in `data/tiles/`, all celestial bodies render using smooth procedural/analytic oblate surfaces with full atmospheric scattering, rings, oceans, and lighting.
- **Baking from Source Maps**:
  1. Download the Solar System source texture pack (or place equirectangular source images into `textures/Solar System/<Body>/<Body>.png|.jpg` and `<Body>_heightmap.png|.tif`).
  2. Run the interactive **Surface Texture Baker** wizard from the repository root:
     ```bat
     .\bake_textures.bat
     ```
     *(Automatically utilizes **CUDA / CuPy** if available, falling back to multi-chunk **NumPy** on CPU. Outputs verified `128×128` oblate-cube-sphere tile packs into `data/tiles/` and updates `data/surfaces.json`.)*
  3. To bake Earth's `16000×8000` cumulus cloud coverage mip pyramid (`textures/Solar System/Earth/clouds/coverage.r8`):
     ```powershell
     python scripts/bake_cloud_maps.py "textures/Solar System/Earth/Earth_clouds.jpg" "textures/Solar System/Earth/clouds"
     ```
- **In-App Drag-and-Drop Baking**: Select any terrestrial planet or moon in the **Inspector**, open the **Surface** tab, and drag-and-drop an equirectangular albedo map or DEM heightmap directly onto the Stellar Forge window to preview and bake tiles live.

---

## Verification, Testing & Benchmarking

### Headless GPU Smoke Test
Verify shader compilation, Vulkan pipeline creation, settings pages, and comparator state transitions headlessly:
```powershell
.\build\bin\StellarForge.exe --self-test
```

### CPU & GPU Contract Test Suite
```powershell
ctest --test-dir build --output-on-failure
```

### Automated Performance Benchmarks
Stellar Forge includes a deterministic benchmark harness with per-subsystem CPU timers, Vulkan hardware timestamp queries, and linear HDR `.pfm` frame capture:
```powershell
# Run standard benchmark scenes (overview, approach, analytical, nbody)
.\build\bin\StellarForge.exe --benchmark

# Run terrain navigation or full audit suites with detailed CSV/JSON output
.\build\bin\StellarForge.exe --benchmark --benchmark-scene terrain-navigation --benchmark-detailed
.\build\bin\StellarForge.exe --benchmark --benchmark-scene saturn --benchmark-variant analytic-shadows --benchmark-capture saturn.pfm
```
Run `.\build\bin\StellarForge.exe --help` for the complete list of benchmark scenes and A/B shader variants.

---

## Repository Architecture

```text
Stellar Forge/
├── CMakeLists.txt          # CMake 3.28+ build config (Vulkan 1.3, GLFW, GLM, ImGui, glslang)
├── run.bat                 # Parallel Ninja build & launch helper
├── bake_textures.bat       # Interactive planetary surface & DEM tile baker launcher
├── project_map.md          # Fast architectural routing index & engineering invariants
├── src/
│   ├── main.cpp            # Entry point & CLI parser
│   ├── app.*               # Window lifecycle, frame loop, UI & dynamic rendering orchestration
│   ├── core/               # Typed units, coordinates, FP64 camera, reverse-Z math, paths, settings
│   ├── ephemeris/          # IAS15 integrator, CSPICE runtime & downloader, Jacobi & Hill hierarchy
│   ├── physics/            # Atmosphere/Mie optics, refraction, eclipse shadows, StarCalculator
│   ├── rendering/          # Celestial, atmosphere, ring, cloud, ocean, bloom, OIT & starfield renderers
│   ├── rhi/                # Vulkan 1.3 context, VMA buffers/images, sync2 barriers, ShaderCache
│   ├── terrain/            # Oblate quadtree LOD, GPU geometry cache, height & albedo tile streaming
│   ├── comparison/         # Object Size Comparator & live multi-system System Comparator
│   ├── ui/                 # Dear ImGui docking workspace, theme & atomic inspector drafts
│   └── benchmark/          # Deterministic camera paths, GPU/CPU telemetry & report generator
├── shaders/                # GLSL 460 compute, vertex & fragment shaders
│   ├── atmosphere/         # Transmittance, multiscatter, sky-view, aerial LUTs & VRS shadow passes
│   ├── celestial/          # Oblate spheres, rings, clouds, orbit lines & Gaia starfield culling
│   ├── common/             # Shared radiative transfer, BRDF, refraction, eclipse & precision math
│   ├── ocean/              # JONSWAP spectrum evolution, 2D FFT & slope/Jacobian moment passes
│   ├── post/               # Multiscale bloom, diffraction spike sprites, ringshine & ACES composite
│   ├── terrain/            # GPU patch evaluation and vertex morphing shaders
│   └── transparency/       # Per-pixel A-buffer clear and O(n log n) resolve compute shaders
├── scripts/                # Python tools for CUDA/CPU tile baking, DEM import & performance audits
├── data/                   # System definitions (Solar System), Gaia DR3 catalog (stars.bin) & configs
├── textures/               # Source planetary textures, ring profiles & blue-noise LUTs
└── docs/                   # 39 deep-dive technical specifications & mathematical derivations
```

For deep-dive mathematical specifications and subsystem architecture notes, explore the [`docs/`](docs/) directory and [`project_map.md`](project_map.md).
