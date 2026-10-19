# Atmospheric shadow deficit

`RenderSettings::atmosphere_shadows_enabled` (Atmospheric shadows in rendering settings, enabled by default) adds full-resolution atmospheric shadows from external bodies and rings. `atmosphere_shadow_samples` selects a strict 4/8/16 integration-sample budget per pixel/star (default 16), shared across primary and nearest-transparent endpoints. It works with both aerial-perspective models (Bruneton and frustum) and hardware blend paths. No atmosphere shadow volume or shadowed sky LUT is allocated.

Atmospheric shadows use the original normalized solar loss fraction architecture:
$$\text{fraction} = \text{clamp}\left(\frac{\sum_s \text{loss}_s}{\sum_s \text{energy}_s}, 0.0, 1.0\right)$$
where each star $s$ evaluates its scattered energy and shadow loss along the visible ray column. In composition, the exact direct solar loss $\Delta L = L \times \text{fraction}$ is subtracted from the clear-atmosphere radiance $L$.

## Mathematical formulation and benefits

1. **Exact zero in full umbra**: In full totality/umbra, $\text{loss}_s \equiv \text{energy}_s$, yielding $\text{fraction} \equiv 1.0$. Because the loss is applied as $L - L \times 1.0 = 0.0$, the discretization error between Riemann sum raymarching and precomputed optical/Bruneton LUTs cancels identically, achieving bitwise 0.0 extinction without light leaks or glowing halos.
2. **Bitwise untouched clear sky**: When unshadowed, $\text{loss}_s = 0.0 \implies \text{fraction} = 0.0$, leaving clear sky untouched.
3. **Independent stellar scattering**: In multi-star systems, unshadowed stars contribute their baseline in-scattered energy to the denominator, ensuring an eclipsed star subtracts only its own direct scattering contribution while leaving other stars completely intact.
4. **Rational observer-extinction proposal**: For camera rays entering the atmosphere, sampling uses a closed-form rational coordinate proposal based on the observer's extinction length:
   $$t = s_{\text{first}} + (s_{\text{last}} - s_{\text{first}}) \cdot \frac{q}{1 + c(1 - q)}$$
   $$ds = (s_{\text{last}} - s_{\text{first}}) \cdot \frac{1 + c}{(1 + c(1 - q))^2 \cdot N}$$
   with $c = 1.5 \cdot \text{dist} \cdot (\bar{\beta}_e)_0$. The exact distance Jacobian $ds$ eliminates numerical root-finding loops and binary searches, achieving accurate integration even at 4 and 16 samples.
5. **Zero temporal lag or ghosting**: Single-frame normalized integration matches the dense 8192-step reference oracle immediately (RMSE < 0.0007 on wide rings, < 0.00006 on thin rings), completely eliminating the need for temporal accumulation, reprojection matrices, or history ping-pong buffers.
6. **Low memory footprint**: By eliminating the 48 B/px ping-pong history textures, the shadow pass requires only two 2-layer textures (`shadow_fraction` at RGBA16F / 8 B/px and `shadow_depth` at R32F / 4 B/px, total 12 B/px), saving over 70 MiB of VRAM at 1080p.

## Candidate culling and intervals

`CelestialRenderer::prepare` supplies the prepared ring hosts and ring extinction atlas. Each body-owned atmosphere table has a separate per-frame shadow SSBO. Body/star candidates use `eclipse_could_shadow` with the entire outer atmosphere as receiver. Ring candidates use the conservative surface predicate with the same outer bound, pixel footprint and prepared ring hosts. Candidates do not depend on caster visibility, hierarchy or mesh contribution. The receiver body itself is excluded because `sun_transmittance` already includes host shadowing. Body and star coordinates are subtracted in double precision before packing atmosphere-relative physical kilometres; oblate atmosphere integration points are transformed back to physical coordinates for shadow tests.

`eclipse_optics.glsl` and `ring_shadow_filter.glsl` accept explicit diagnostic/filter flags and are shared with the mesh path. Rings retain the approximate single-box penumbra, including stellar angular radius, distance-dependent softness, grazing stretch and pixel filtering. The 4097-by-64 prefix extinction table describes the ring profile and its illumination angles; it does not quantize shadow positions into atmosphere cells.

A full-resolution compute pass produces two prefix corrections: layer 0 for the opaque/sky endpoint and layer 1 for the nearest transparent fragment. Transparent fragments beyond the first use the analytic LUT baseline.

## Verification and cost

```powershell
.\build\StellarRenderingTests.exe --atmosphere-shadows scratch/atmosphere-shadows/normalized
.\build\StellarRenderingTests.exe --atmosphere-shadow-benchmark scratch/atmosphere-shadows/benchmark-normalized.json 16
```

The focused test suite tests:
- Independent 32768-step double-precision single-scattering oracle checks at various caster offsets.
- Full-umbra ambient suppression with zero residual leak.
- Finite opaque and transparent endpoints in both Bruneton and frustum aerial models.
- Overlapping umbrae direct radiance subtraction.
- Multi-star independent stellar scattering preservation.
- Full-resolution own-ring atmosphere shadow contracts at orbital resolution (>32,000 affected pixels).
- Ring deficit RMSE vs. dense unclipped reference (< 0.0007 on wide rings, < 0.00006 on thin rings) on the first frame.
- Unchanged optical and Bruneton cache tables across caster motion.

RTX 4060 Ti Release benchmark timings (1920x1080, 16 samples):

| Synthetic view | Shadows off composite (ms) | Shadows on composite (ms) | GPU shadow pass (ms) | Net shadow cost (ms) |
| --- | ---: | ---: | ---: | ---: |
| Clear close-up (Bruneton) | 0.194 | 0.217 | 0.000 | 0.023 |
| Eclipse close-up (Bruneton) | 0.189 | 2.681 | 2.355 | 2.492 |
| Orbital ring shadow (Bruneton) | 0.116 | 0.808 | 0.653 | 0.692 |

Performance comparison against legacy stochastic absolute-deficit method:
- **Ring shadows**: GPU shadow cost dropped from **17.58 ms** down to **0.65 ms** (**27x speedup**).
- **Close eclipses**: GPU shadow cost dropped from **6.82 ms** down to **2.35 ms** (**2.8x speedup**).
- **Memory footprint**: Reduced by **75%** (from 48 B/px to 12 B/px).
- **Stability**: Zero ghosting, zero history smearing during translation, and instantaneous single-frame convergence.
