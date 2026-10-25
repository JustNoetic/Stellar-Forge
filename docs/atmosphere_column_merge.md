# Combined sky and aerial column integration

25 October 2026.

When aerial perspective is enabled, `aerial_perspective.comp` now writes the final cumulative radiance and `exp(-tau)` to the existing sky images. `AtmosphereRenderer::prepare` dispatches this combined bake instead of dispatching both sky and aerial marches. Aerial-disabled rendering retains the standalone sky shader.

The angular mapping, 192 × 108 grid, 63 intervals, midpoint/subdivision rules, optical tables and image allocations are unchanged. The merge removes one duplicated column march, with the upper bound of 2,612,736 medium-cell evaluations per ordinary update before empty-ray rejection and detached-layer subdivisions. No additional images or retained memory are needed.

The combined bake increments both product update counters and is timed under `atmosphere_aerial`. `atmosphere_sky` is recorded only for the standalone sky path. The existing aerial-valid flag preserves an unchanged volume across toggles and invalidates it after movement/lighting/optical changes while aerial perspective is disabled. A refresh of that stale volume also refreshes the sky images from the same accumulator.

## Validation

- Release application and rendering-test builds passed.
- `--atmosphere-columns-only`: 4,644,864 half-float components matched the standalone sky shader bit for bit across 28 cases. Coverage includes vacuum, molecular/Mie scattering, detached aerosols, giant radii, two differently colored stars, negative altitude, near-floor views, both sides of the atmosphere boundary and orbital columns.
- `--atmospheres-only`: passed with Vulkan validation enabled, including atmospheric shadows, opaque/transparent endpoints, physical/particle/giant atmospheres, cache pressure, RGB extinction, negative altitude, oblate geometry, horizon continuity, exposure and rotation/FOV reuse. Added assertions cover unchanged toggle reuse, standalone sky updates while aerial perspective is disabled, and stale-volume refresh on re-enable.
- Application `--self-test`: completed three frames cleanly with Vulkan validation enabled. No validation errors were reported by these runs.

## GPU timing

RTX 4060 Ti, Release, 1920 × 1080. The existing `--atmosphere-benchmark` was captured before and after the merge. It uses serialized offscreen rendering with three repetitions, alternating floor order, and 8 warmup + 24 measured frames per case. Timings exclude readback and presentation; these are atmosphere preparation intervals, not whole-frame timings or application FPS. Validation was disabled for both timing captures.

Representative medians, with the -10.984 km terrain floor:

| View / change | Before, ms | Combined, ms | Reduction |
| --- | ---: | ---: | ---: |
| Ground / camera translation | 0.222208 | 0.113664 | 48.8% |
| Orbit / camera translation | 0.242688 | 0.123904 | 48.9% |
| Ground / optical update | 0.279552 | 0.167936 | 39.9% |
| Orbit / optical update | 0.293888 | 0.176128 | 40.1% |

Cached and rotation-only cases retain zero median LUT preparation time. Orbital translation steps are close to float-altitude quantization, so some frames reuse the cache: pooled mean preparation time in that case is 0.121870 → 0.062379 ms. Optical-update cases rebuild every measured frame and include unchanged optical-table work. Startup/clock tails in the initial cached cases make tiny composition differences unsuitable for attributing savings to this change.

Artifacts are retained under the project's ignored `tests/atmosphere-column-merge/` directory: [before timings](../tests/atmosphere-column-merge/before.json), [after timings](../tests/atmosphere-column-merge/after.json), [comparison](../tests/atmosphere-column-merge/comparison.json), and validation logs. The direct differential test is `tests/atmosphere_column_tests.hpp`, invoked from the rendering-test driver and included in `--atmospheres-only`.
