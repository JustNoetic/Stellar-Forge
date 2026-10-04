# Ground rotation and N-body warp fixes

Implemented in `D:\Files\Coding\OpenGL\Stellar-Forge`. The CPU terrain LOD path remains active.

## Changes

- Terrain and cloud vertices subtract a split body-local camera origin before spin/world transforms. Large planet-radius coordinates no longer cancel after rotation in GPU float arithmetic.
- Ground-follow rotation accepts all angular steps, including the steps previously rejected at 0.5 radians. CPU surface queries use the same full-precision pole as camera following.
- Clearance and approach queries probe the camera's predicted position after rotation. They no longer query a different longitude with the previous camera pose. Exact repeated surface queries are shared only within one render frame, after tile uploads.
- The N-body pair-collision scan uses compiled code that releases Python's GIL, preserving the previous first-pair merge ordering and strict overlap condition.
- Corrected a missing warmup argument and moved both physics-thread launches after warmup. The previous error aborted integrator warmup and left compilation competing with rendering.

## Verification

Actual NVIDIA RTX 4060 Ti, production terrain transform, a fixed point and camera one metre above it, 8,192 rotation angles:

| Measurement | Previous | Fixed |
|---|---:|---:|
| Maximum rotation-dependent position error | 4.03 m | 0.00000060 m |
| RMS position error | 1.70 m | 0.00000023 m |

These measurements isolate the rotation transform. They do not assert that all other terrain, texture, shading, or LOD artifacts have been eliminated.

The 56-body collision scan dropped from approximately 1.95 ms to 0.003 ms per call. 204 randomized/boundary cases preserve collision ordering. Regression checks also verify that exact repeated surface queries reuse work and moved queries reevaluate it. Six existing terrain geometry-cache GPU tests passed.

Hidden 512×512 application test, existing graphics settings, landed Earth, 56-body N-body simulation at 8,192× (2.28 simulated hours/second), 180 measured warp frames after warming terrain residency:

| Frame-time measurement | Previous | Final |
|---|---:|---:|
| Median | 9.13 ms | 5.14 ms |
| 95th percentile | 10.73 ms | 6.44 ms |
| Maximum observed | 24.69 ms | 16.89 ms |

This is a controlled scene, not a prediction for another resolution or camera location. Instrumentation includes GPU completion at presentation. Some frame outliers remain. Graphics settings hashes were unchanged after the application runs.

## Ring shadows

A temporary ringed-Earth scene was also exercised through a full rotation without modifying the saved system. Mode 3 still switches shadow-intersecting atmospheric rays to numerical integration; this change preserves that rendering behavior. Rotating into and out of a ring shadow therefore still changes GPU workload. The heavier shadow fallback needs a separate optimization; this patch does not claim to eliminate its FPS dip.

## Reproduce

Run the engine's Python with `scripts/test_ground_warp.py` and `scripts/test_terrain_geometry_cache.py`. Restart the engine to load the changes. The first launch may take longer while Numba prepares its compiled functions before rendering begins.
