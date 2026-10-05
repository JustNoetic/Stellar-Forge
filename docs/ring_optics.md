# Ring optical response

Visible rings, host ringshine maps, and CPU ringshine use the same optical contract.
GLSL lives in `common/ring_optics.glsl`; its CPU counterpart is
`engine/rendering/ring_optics.py`. `common/ring_source.glsl` supplies source quadrature.

## Parameters and transport

- `alpha` is vertical extinction opacity, never camera-path opacity. It is clamped
  to `[0, 1 - 2^-23]`, then converted continuously to `tau = -log(1-alpha)`.
- `asymmetry` and `backscatter` are lobe shape parameters, both clamped to ±0.99.
  The UI uses the same limits. Saved values outside that range are clamped at
  evaluation. The second parameter may be positive, giving two forward lobes.
- Phase cosine is incoming photon travel dotted with outgoing travel: `-dot(L,V)`.
  HG and CS each integrate to one over solid angle. CS's g is a shape parameter,
  not exactly its mean scattering cosine.
- Textured layers use HG with chunk fraction `clamp((alpha-.1)/.5,0,1)`. Their
  forward weight is `.95-.45*chunks`; their multiple weight is `chunks`.
- Procedural layers use CS with forward weight `.1+.8*clamp(scatter,0,1)` and
  multiple weight `clamp(1-.7*clamp(scatter,0,1),.1,1)`.
- Lit-face slab response is `mu0/(mu_v+mu0)*(1-exp(-tau/mu_v-tau/mu0))`.
  Opposite-face response uses the stable exponential-difference ratio, including
  its equal-depth limit. Small arguments use series in GLSL and `expm1`/`log1p`
  on CPU. There is no artificial illumination at zero source elevation.
- The existing isotropic multiple-scattering approximation uses albedo .92 and
  Hapke H functions. The existing opposition approximation multiplies lit-face
  single scattering. These authored terms now apply consistently in all paths.

Compositing opacity still uses the view path. Radiance already includes extinction
and uses premultiplied blending; it must not be multiplied by opacity a second time.
The engine's direct radiance convention includes a factor of pi.

## Overlapping materials

The model treats coplanar layers as a homogeneously mixed slab. At each radial
sample, total extinction depth is the sum of each layer's vertical tau. Each
layer evaluates its own HG/CS mixture using its own alpha and parameters;
responses are combined with `tau_layer/tau_total` weights and linear RGB color.
Neither asymmetry nor material family is averaged before optical evaluation.
This is not a vertically ordered stack of physically separated layers.

The texture atlas retains 16 unified extinction rows and 256 original segment rows.
The host optical bake samples segment rows and receives each layer's geometry and
properties directly. The obsolete averaged phase-property atlas is removed.
Texture edits and tint are applied once in sRGB before linearization, matching
the segment profiles used by secondary lighting.

CPU illumination of distant moons retains an annulus geometry approximation,
but evaluates material responses at 64 Gaussian radial-area nodes before summing.
Compact samples preserve each layer's phase, color, and shared extinction.
`build_secondary_ring_properties` returns four arrays; `params[body,3]` is now
a one-based slot into the compact optical samples when those samples are supplied.

## Extended sources

A uniform-radiance circular disk uses four Gaussian nodes in projected radius
squared and sixteen symmetric azimuth nodes: 64 samples. The weight includes
the `dOmega` Jacobian. Stars crossing the ring plane illuminate both faces, each
with its sampled signed elevation and phase. Smaller stars away from the plane
use the point limit when `sin(radius)<=.001` and `abs(sun)>=2*sin(radius)`.
This threshold and quadrature are shared by visible rings, host maps, and CPU.

Host-map keys include stellar angular radius (six significant digits) as well as
material revision, geometry, band count, and sine elevation. A stellar radius
change invalidates the map; camera movement and star azimuth still reuse it.

Planetshine uses the same optical response without peak renormalization. Disk
rays intersect the actual oblate host; each visible surface point contributes
its Lambertian daylight incidence and sampled solid angle. The host's Lambert
`1/pi` cancels the ring's engine convention `pi`. A hidden day side contributes
zero through geometry, without using the ring point's solar shadow as a gate.

Source integrals remain finite quadrature, not exact integrals. Very narrow lobes,
large sources, ringlet profiles, or near-contact geometry can require more samples.
Direct-light caster penumbrae retain the existing analytic approximation. The
host is a Lambertian reflector for planetshine; secondary eclipses, terrain, and
Hapke host reflection are not integrated in that source model. Normalized phase
lobes do not by themselves establish energy conservation of the authored
multiple-scattering/opposition model.

## Verification

Run `python scripts/test_ring_optics.py` with NumPy, Numba, ModernGL, and an OpenGL
4.6 driver. The script uses EGL on Linux and writes results under
`scratch/ring-optics-tests/`. `STELLAR_FORGE_ROOT` and `RING_TEST_OUTPUT` can override
source and output directories for staged verification.

The suite compiles the production programs; checks GPU/CPU optical parity,
normalization, dense-opacity continuity, both faces, mixed material families,
finite-star continuity, independent dense-source quadrature, oblate planetshine,
the actual compiled Numba moon-lighting entry point, atlas assembly, and cache
invalidation. GPU timings cover a synthetic 1024×256 ring pass, not whole-app FPS.
