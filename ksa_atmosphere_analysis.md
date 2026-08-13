# Kitten Space Agency (KSA) Atmosphere Rendering Architecture Analysis

This document provides a comprehensive breakdown of the atmosphere rendering architecture used in Kitten Space Agency (KSA), based on the `Atmosphere` and `Godrays` shader directories. The goal is to provide a blueprint for porting this high-performance architecture into Stellar-Forge (SF).

## 1. Architectural Overview

KSA uses a **precomputed Look-Up Table (LUT)** approach inspired by Bruneton's and Hillaire's scalable atmosphere models. 

Instead of raymarching the atmosphere mathematical equations (Rayleigh, Mie, Transmittance) for every pixel on the screen during the main rendering pass (which is what SF currently does in `atmo.frag`), KSA decouples the complex volumetric math from the screen resolution. It evaluates the atmosphere into low-resolution LUTs using compute shaders, and then samples these LUTs in a lightweight fullscreen composition pass.

This splits the atmosphere pipeline into two distinct phases:
1. **Precomputation Phase:** Generating the LUTs.
2. **Rendering Phase:** Compositing the LUTs to the screen and evaluating dynamic volumetric shadows (like eclipses and godrays).

---

## 2. The Precomputation Phase (The LUTs)

KSA relies on four primary compute shaders to generate its LUTs. These are updated as needed (some are static, some update per-frame based on camera movement).

### A. Transmittance LUT (`TransmittanceLut.comp`)
- **What it does:** Calculates how much light makes it through the atmosphere from any given altitude and sun angle.
- **How it works:** It raymarches from a given point out to the edge of the atmosphere, integrating Rayleigh, Mie, and Ozone absorption. 
- **The "Soft Shadow" Trick:** KSA artificially adds a planetary self-shadow penumbra (the terminator line) when reading from this LUT using a hardcoded `0.01` radian sun size via `smoothstep`. 
- **SF Adaptation Note:** For SF, you should replace KSA's hardcoded `0.01` with a dynamic `starRadius / distanceToStar` uniform to accurately handle massive or very close stars.

### B. Multiple Scattering LUT (`MultipleScatteringLut.comp` & `AmbientLut.comp`)
- **What it does:** Approximates the light that bounces multiple times inside the atmosphere (without this, the atmosphere looks too dark).
- **How it works:** It uses an iterative approach or a dual-scattering approximation. It heavily relies on the Transmittance LUT.

### C. Sky LUT (`SkyLut.comp`)
- **What it does:** A 2D texture (typically mapped using spherical or angular coordinates) that stores the color of the *infinite sky dome* (rays that do not hit terrain).
- **Why it's fast:** If a pixel on the screen is looking at the sky (depth == 1.0), KSA does not raymarch. It just performs an $O(1)$ 2D texture fetch from the Sky LUT. 

### D. Aerial Perspective LUT (`AerialPerspectiveLut.comp`)
- **What it does:** A 3D texture (frustum voxel grid) that stores the inscatter and transmittance for points *inside* the atmosphere (e.g., between the camera and a mountain, or a ship in orbit).
- **How it works:** It raymarches the atmosphere along the camera's view frustum. The $Z$-axis of the 3D texture corresponds to depth slices (non-linear to give more resolution near the camera).
- **Why it's fast:** Like the Sky LUT, this decouples volumetric raymarching from the screen resolution. A 1080p, 1440p, or 4K screen all evaluate the exact same 32x32x32 (or similar) 3D grid.

---

## 3. The Rendering Phase

Once the LUTs are generated, KSA uses `Atmosphere.comp` to draw the atmosphere to the screen. 

### Base Atmosphere Evaluation (Lightning Fast)
1. KSA checks the depth buffer.
2. If the ray hits infinity (the sky), it samples the 2D **Sky LUT**.
3. If the ray hits geometry (terrain/ships), it samples the 3D **Aerial Perspective LUT** using trilinear (or bicubic) interpolation based on the distance to the geometry.
4. *Result:* The unshadowed atmosphere is calculated in a fraction of a millisecond.

### Dynamic Volumetric Shadows (Eclipses & Rings)
Because LUTs assume radial symmetry and a clear sky, they cannot store local shadows (like an eclipse or a ring shadow). KSA solves this by evaluating shadows dynamically *on top* of the LUTs.

Inside `Atmosphere.comp`, KSA runs `GetVolumetricEclipsesShadowedInscatter()`:
1. It detects if the current pixel ray intersects a shadow volume (like a moon casting an eclipse).
2. If it does, it runs a short, fast raymarching loop (e.g., 15 steps) *only* inside the shadowed region.
3. It calculates exactly how much light was blocked (the "shadowed inscatter").
4. It **subtracts** this blocked light from the base "clear sky" color retrieved from the LUT.

This is how KSA maintains **pixel-perfect, raymarched volumetric shadows** without the immense cost of raymarching the entire atmosphere.

---

## 4. How to Adapt this to Stellar-Forge (SF)

To migrate SF from its current brute-force per-pixel raymarching (`atmo.frag`) to KSA's architecture, you should follow this roadmap:

### Step 1: Shift to Compute Shaders
Move your atmosphere rendering out of `atmo.frag` and into Compute Shaders. Compute shaders are infinitely better for generating 3D textures (Aerial Perspective) and managing volumetric workloads.

### Step 2: Implement the LUT Generation
1. Port KSA's `TransmittanceLut.comp` and Multiple Scattering logic.
2. Implement the **Sky LUT (2D)**. This alone will save immense frame time when looking at the sky.
3. Implement the **Aerial Perspective LUT (3D)**. Map it to your camera's frustum.

### Step 3: Implement the Screen-Space Composition Pass
Replace your heavy raymarching loop in `atmo.frag` (or create a new composition compute shader) with simple texture fetches:
- `color = texture(SkyLUT)` for sky pixels.
- `color = texture(AerialPerspectiveLUT)` for terrain/object pixels.

### Step 4: Port Your Shadows (Rings & Eclipses)
You already have an excellent analytical shadow function in SF (`compute_shadow`). 
1. Use your new composition pass.
2. When a pixel is inside a ring shadow or eclipse, run a short raymarching loop using `compute_shadow`.
3. Calculate the light blocked by the shadow, and subtract it from the LUT color.

### Step 5: Fix the Terminator Penumbra
In your new Transmittance LUT sampling function, remember to replace KSA's hardcoded sun angular radius with a dynamic one to perfectly support SF's massive/close stars:
```glsl
// Do this:
float dynamicSunAngularRadius = starRadius / distanceToStar;
float planetSoftShadow = smoothstep(-sinHorizonAngle * dynamicSunAngularRadius, sinHorizonAngle * dynamicSunAngularRadius, cosZenith - cosHorizonAngle);
// Instead of KSA's hardcoded: float sunAngularRadius = 0.01;
```

## 5. Achieving a Pixel-Perfect Limb

One of the biggest challenges with low-resolution precomputed LUTs is achieving a perfectly smooth, anti-aliased horizon (the limb) where the planet meets space. Bilinear filtering typically causes horrible bleeding and blocky artifacts here. KSA employs four specific techniques to solve this:

1. **Horizon-Aligned Non-Linear Mapping:** In `AtmosphereLuts.glsl`, the vertical axis ($Y$) of the LUT is mapped mathematically to the *angle relative to the horizon*, forcing the horizon line to sit exactly at `Y = 0.5`. A `sqrt()` function is then applied to cram the majority of the texture resolution right into the few degrees immediately surrounding this horizon line.
2. **Anti-Bleed Texel Snapping:** When sampling the LUT in `GetAerialPerspectiveColorFromLuts()`, the shader intercepts the UVs. If a pixel is dangerously close to the `0.5` horizon line, it snaps the UV strictly into "space" or "ground" and clamps the sampler. This physically prevents the GPU's bilinear filtering from blending the dark planet surface into the bright sky.
3. **Custom Bicubic Filtering:** Instead of hardware trilinear filtering (which causes blocky grid artifacts when stretching a low-res volumetric LUT), KSA uses a custom `SampleAerialPerspectiveBicubic()` function to curve the interpolation, completely hiding the LUT's low resolution.
4. **Analytical Depth Snapping:** At orbital distances, standard depth buffers lose precision, causing the 3D planet mesh to look jagged against the sky (z-fighting). In `Atmosphere.comp`, KSA completely bypasses the 3D mesh when rendering the atmosphere. It mathematically calculates the perfect ray-sphere intersection (`planetIntersections.x`) and snaps the pixel's depth directly to this mathematical sphere, making the physical edge of the planet flawlessly smooth down to the sub-pixel level.

## Summary
By adopting this architecture, SF will retain its highly accurate, mathematically rigorous shadow calculations (like ring penumbras and refraction), but will evaluate the baseline atmosphere at a fraction of the performance cost by caching it in LUTs, all while maintaining a flawlessly smooth planetary limb.
