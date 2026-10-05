# Ringshine: ring-to-host surface illumination

This document explains Stellar Forge's host-planet ringshine implementation: starlight scatters in a planetary ring, then illuminates the planet's surface. The ring acts as an extended, directionally scattering source. The renderer integrates that source into a reusable surface illumination map rather than evaluating the full ring integral at every visible fragment.

Ringshine adheres to the unified optical contract defined in [Ring optical response](ring_optics.md), implemented on GPU in [`common/ring_optics.glsl`](../engine/glsl/common/ring_optics.glsl) (with extended source quadrature in [`common/ring_source.glsl`](../engine/glsl/common/ring_source.glsl)) and on CPU in [`engine/rendering/ring_optics.py`](../engine/rendering/ring_optics.py). The obsolete averaged phase-property atlas was removed: the host optical bake samples original segment rows directly from [`u_ring_gradients`](../engine/rendering/render_utils.py) and receives each layer's geometry and properties via uniform arrays, evaluating coplanar layers as a homogeneously mixed slab. Furthermore, the bake integrates over finite circular stellar disks when crossing the ring plane, maintaining continuous lighting across equinox.

**Scope:** the ring-to-host surface path, including ring material inputs, optical transfer, oblate geometry, numerical integration, map storage, caching, and surface lookup. Rendering the visible rings is covered only where it supplies inputs or explains a shared contract. Illumination of moons, atmospheric transport, and cloud shading are outside this document.

**Source baseline:** production ring optics architecture following the unified optical contract. Equations below transcribe the implementation; they are not a proposal for a different renderer. “Exact” geometry means exact within the aligned spheroid, flat ring, and extended circular-star disk model. The integral and its stored reconstruction remain numerical approximations.

## 1. Technique overview

There are two different visibility questions for each ring element:

1. Can the surface point see it above the planet's local horizon?
2. Does the element receive sunlight, or is it inside the planet's shadow?

For every ring element satisfying both conditions, the bake evaluates its radiance toward the surface. That radiance depends on vertical optical depth, viewing elevation, solar elevation, scattering angle, individual layer materials, and which face of the ring the surface sees. The contribution is then weighted by the element's projected solid angle and the surface's incidence cosine.

~~~mermaid
flowchart TD
    A["Segment profiles in atlas (rows 16+) + segment uniforms"] --> D["Host/star map bake"]
    B["Host shape, star elevation, star angular radius"] --> D
    D --> E["Extended source quadrature (1 or 64 samples)"]
    E --> F["Clip horizon-visible arcs against host shadow"]
    F --> G["Integrate mixed-slab radiance × projected solid angle"]
    G --> H["Cached 128 × 65 RGB transfer tile"]
    H --> I["Surface lookup by latitude and anti-solar azimuth"]
    J["Current stellar color and flux"] --> K["Diffuse surface contribution"]
    I --> K
    L["Surface albedo"] --> K
~~~

The key reduction is rotational symmetry. With a radially varying ring and an axisymmetric host, a fixed star elevation and stellar angular radius produce a two-dimensional illumination field over the host. Changing stellar azimuth only rotates that field. Changing stellar brightness scales it. Neither change requires integrating the ring again.

This is deterministic quadrature into a cached texture. It is not screen-space illumination, a Monte Carlo path tracer, or a constant ambient tint.

## 2. Implementation map

Paths below are relative to the repository.

| Responsibility | Source and symbols |
| --- | --- |
| Texture allocation, dirty keys, per-host/per-star bake scheduling | [ringshine.py](../engine/rendering/ringshine.py): `RingshineMap.__init__`, `update`, `release`, `build_secondary_ring_properties` |
| Per-texel surface coordinates, radial quadrature, segment sampling, extended source loop | [ringshine_map.frag](../engine/glsl/post/ringshine_map.frag): `main`, `ringshine_material_radiance`, `u_segment_geometry`, `u_segment_props`, `u_segment_textured` |
| Oblate horizon/shadow clipping, warped azimuth quadrature | [ringshine_integral.glsl](../engine/glsl/common/ringshine_integral.glsl): `ringshine_arc`, `ringshine_band` |
| Shared optical response: slab transfer, phase functions, opposition surge | [ring_optics.glsl](../engine/glsl/common/ring_optics.glsl): `ring_tau`, `ring_hg`, `ring_cs`, `ring_phase`, `ring_transfer`, `ring_opposition`, `ring_radiance` |
| Extended stellar disk quadrature (64 Gaussian/azimuth nodes) | [ring_source.glsl](../engine/glsl/common/ring_source.glsl): `ring_disk_sample`, `ring_resolve_disk`, `RING_DISK_SAMPLES` |
| Tile-safe bilinear lookup | [ringshine_lookup.glsl](../engine/glsl/common/ringshine_lookup.glsl): `ringshine_sample` |
| Full-screen bake vertex shader | [ringshine_map.vert](../engine/glsl/post/ringshine_map.vert) |
| Radial profile preparation and host atlas rebuilds | [render_utils.py](../engine/rendering/render_utils.py): `generate_ring_shadow_grad`, `bake_unified_shadow_profile_jit`, `rebuild_ring_gradients_atlas` |
| Runtime geometry, map updates, texture binding | [app.py](../engine/app.py): ring-buffer construction and `self.ringshine_maps.update(...)` |
| Surface receivers | [sphere.frag](../engine/glsl/celestial/sphere.frag), [terrain.frag](../engine/glsl/celestial/terrain.frag): `// === Ringshine ===` blocks |
| CPU counterpart of the optical response and moon illumination | [ring_optics.py](../engine/rendering/ring_optics.py): `ring_tau`, `ring_hg`, `ring_cs`, `ring_phase`, `ring_transfer`, `ring_phase_radiance`, `ring_mixed_radiance` |
| Oblate planetshine transport using shared ring optics | [planetshine.py](../engine/rendering/planetshine.py): `compute_planetshine_ring` |
| Optical regression suite | [test_ring_optics.py](../scripts/test_ring_optics.py) |
| User controls and persistence | [modals.py](../engine/ui/modals.py), [input_handler.py](../engine/core/input_handler.py) |

The host map is integrated on the GPU via `ringshine_map.frag` calling `ring_optics.glsl` and `ring_source.glsl`. The CPU functions in `ring_optics.py` maintain mathematical parity and evaluate the distant-moon lighting integral. The obsolete property atlas `u_ring_props` was removed; segment properties are passed directly via uniform arrays.

## 3. The transport integral and its normalization

### 3.1 From an extended ring to surface irradiance

Let $P$ be the receiving surface point, $Q$ a ring element, $d=\lVert Q-P\rVert$, and $\omega=(Q-P)/d$. Define:

- $\mu_p=\max(0,N_p\cdot\omega)$: incidence cosine at the planet.
- $\mu_v=|N_r\cdot(-\omega)|$: projected ring-area cosine toward the planet.
- $L_r(Q\rightarrow P; S)$: outgoing ring radiance toward the receiver under incident starlight along unit direction $S$.
- $V(P,Q,S)$: visibility, including both the host horizon and sunlight reaching the ring from direction $S$.

The ring contribution has the area-integral form

$$
E_r(P)=\int_{\mathrm{ring}}
\int_{\Omega_\star}
L_r(Q\rightarrow P; S)\,
V(P,Q,S)\,
\frac{\mu_p\mu_v}{d^2}\,
\frac{d\Omega_\star(S)}{\pi\sin^2 r_\star}
\,dA.
$$

The $\mu_v/d^2$ factor converts source area to solid angle; $\mu_p$ converts incident radiance to receiver irradiance. This is the standard extended-source change of integration measure described in [PBRT's Working with Radiometric Integrals](https://pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Working_with_Radiometric_Integrals).

For a flat annulus, $dA=r\,dr\,da$, where $a$ is ring azimuth. The implementation evaluates a color-valued, unit-stellar-input transfer

$$
K(P)=
\int_{r_{\mathrm{first}}}^{r_o}
\sum_{q=1}^{N_\star} w_{\star,q}
\int_{\mathcal A(r,P,S_q)}
L_r(Q\rightarrow P; S_q)
\frac{\mu_p\mu_v}{d^2}\,r\,da\,dr.
$$

Here $N_\star$ is 1 for point stars or 64 for resolved finite stellar disks, $w_{\star,q}$ is the source quadrature weight, $S_q$ is each sampled ray from the stellar disk, and $\mathcal A(r,P,S_q)$ contains only visible, illuminated arcs for that source sample. All geometry is normalized to the host's equatorial radius.

At each radial position $r$, coplanar layers are treated as a homogeneously mixed slab with total normal optical depth $\tau_{\mathrm{total}}=\sum_i \tau_i(r)$. Outgoing radiance combines each layer's response weighted by its optical depth fraction:

$$
L_r(Q\rightarrow P; S_q)=
\sum_{i=1}^{M}
C_{i,\mathrm{lin}}(r)\,
\frac{\tau_i(r)}{\tau_{\mathrm{total}}(r)}
\left[
S_{\mathrm{lit/opp},i}\,p_i\,\Omega_i
+ S_{\mathrm{ms}}\,p_{\mathrm{ms},i}
\right].
$$

Here $C_{i,\mathrm{lin}}=\max(C_{i,\mathrm{profile}},0)^{2.2}$, $p_i$ is layer $i$'s phase function, $p_{\mathrm{ms},i}$ is its multiple-scattering weight, $S_{\mathrm{lit/opp},i}$ and $S_{\mathrm{ms}}$ are the slab transfer terms, and $\Omega_i$ is the opposition multiplier on the lit face or $\max(0,\mathrm{unlit\_factor}_i)$ on the unlit face.

**There is no extra solar cosine outside $L_r$.** The slab response already includes illumination and attenuation as functions of solar elevation.

### 3.2 What the texture actually stores

The bake ends with:

~~~glsl
out_color = vec4(total * RS_PI, 1.0);
~~~

Thus the stored RGB is

$$
M=\pi K.
$$

The surface shaders apply `0.318309886`, approximately $1/\pi$, when reading it:

$$
C_{\mathrm{ringshine}}=
A_{\mathrm{surface}}
\odot
\sum_j
\left[
\frac{M_j}{\pi}
\odot C_{\star,j}
\right]F_j
=
A_{\mathrm{surface}}
\odot
\sum_j
\left[
K_j\odot C_{\star,j}
\right]F_j.
$$

$\odot$ denotes componentwise RGB multiplication. $F_j$ is the runtime stellar falloff:

$$
F_j=
\begin{cases}
\mathrm{starLum}_j/\max(D_j^2,10^{-8}), & \text{HDR enabled},\\
1, & \text{HDR disabled}.
\end{cases}
$$

Although shader variables call the sample “irradiance,” the texture is specifically a **π-scaled normalized transfer map**, not an absolute irradiance measurement in physical units. The bake's π and the consumer's $1/\pi$ cancel in the implemented result. This fits the renderer's diffuse color convention; it should not be silently replaced by a differently normalized radiometric quantity.

The map already includes ring color, layer opacities, surface incidence cosine, and projected solid angle. It excludes surface albedo, stellar RGB, stellar luminosity/falloff, and exposure.

Consequently, a consumer must not multiply it again by ring area, ring opacity, $\mu_p$, $\mu_v$, or solar elevation.

## 4. Coordinates, units, and symbols

The CPU supplies render-space positions and ring poles. As indexed by [project_map.md](../project_map.md), physics coordinates $(x,y,z)$ become render coordinates $(x,z,-y)$. Geometry passed to the bake must already use a consistent frame.

Runtime ring and host radii share the renderer's AU units. Before integration, `RingshineMap.update` divides the inner and outer ring radii by the host's equatorial radius. All distances in the integral are therefore dimensionless.

| Symbol | Meaning | Code |
| --- | --- | --- |
| $r_i,r_o$ | Ring bounds in host equatorial radii | `u_host_params.xy` |
| $f$ | Host flattening, clamped to $[0,0.95]$ | `u_host_params.w` |
| $q=1-f$ | Polar/equatorial radius ratio | `ff` |
| $\eta=q^{-2}$ | Spheroid coefficient | `eta` |
| $\lambda$ | Geocentric latitude relative to the ring plane | Represented by `signed_slat` |
| $\ell=\sin\lambda$ | Signed radial-direction elevation | `signed_slat` / `frag_elevation` |
| $s=|\ell|,\ c=\sqrt{1-s^2}$ | Folded latitude sine and cosine | `slat`, `clat` |
| $\rho$ | Host surface radius at that geocentric latitude | `rho` |
| $b=\sin B$ | Signed stellar elevation above the ring plane | `u_sun_elevation` / `sun` |
| $r_\star$ | Stellar angular radius; $\sin r_\star$ clamped to $[0,0.999]$ | `u_sun_radius` |
| $N_\star$ | Source quadrature sample count (1 point or 64 disk) | `sources` |
| $w_{\star,q}$ | Quadrature weight of stellar disk sample $q$ | `source.w` |
| $\mu_0=|b_q|$ | Absolute incidence cosine at the ring for sample $q$ | `abs(sun)` / `mu_0` |
| $\phi$ | Surface azimuth relative to the projected anti-solar direction | `phi_center` / `phi` |
| $a$ | Integration azimuth relative to the surface meridian, with shadow centered at $\phi$ | `a` |
| $\chi=\mu$ | Phase cosine $-\operatorname{dot}(L,V)$; not an angle in radians | `theta` / `mu` |
| $\alpha_i$ | Normal vertical extinction opacity of layer $i$, clamped to $[0,1-2^{-23}]$ | `layer_alpha[i]` |
| $\tau_i$ | Normal optical depth of layer $i$, $-\ln(1-\alpha_i)$ with small-alpha series | `layer_tau[i]` |
| $\tau_{\mathrm{total}}$ | Summed vertical normal optical depth of all active layers at radius $r$ | `material_tau` |
| $g_f,g_b$ | Forward and backward lobe shape parameters, clamped to $\pm 0.99$ | `u_segment_props[i].xy` |

Latitude is **geocentric**, obtained from the normalized position relative to the host center. It is not the latitude of the spheroid's surface normal. The integral separately constructs the correct oblate normal.

Uniform rescaling of the host and ring leaves $K$ unchanged: physical source area contributes a factor $R_{\mathrm{eq}}^2$, and distance squared contributes its inverse. Actual star-distance changes still affect $F_j$ during shading.

## 5. Ring material inputs and homogeneous slab mixing

The visible ring is rendered separately by [ring.frag](../engine/glsl/celestial/ring.frag). Ringshine does not read its screen image or rasterized brightness. It uses radial material profiles shared with the ring/shadow machinery and evaluates the shared optical response.

### 5.1 Segment profiles and atlas layout

`generate_ring_shadow_grad` builds a 4096-sample RGBA profile per segment:

- An opacity gradient supplies radial alpha, or alpha defaults to one.
- For a textured segment, processed texture RGB is multiplied by its ring color; texture alpha is multiplied by the radial opacity gradient.
- For an untextured segment, RGB is the supplied ring color and alpha comes from the gradient.
- Texture cosmetic adjustments and tints are applied once in sRGB before linearization: $C_{\mathrm{lin}}=\max(C_{\mathrm{profile}},0)^{2.2}$.

The texture atlas [`u_ring_gradients`](../engine/rendering/render_utils.py) has dimensions 4096 × 272 in RGBA float32:

| Rows | Contents | Usage |
| --- | --- | --- |
| 0–15 | Unified host extinction profiles | Used for fast shadow queries across the host |
| 16–271 | Original per-segment profiles (up to 256 rows) | Sampled directly by visible rings, host ringshine, and CPU lighting |

**The obsolete averaged phase-property atlas (`u_ring_props`) is removed.** Phase parameters and material families are no longer blended or stored into intermediate texture rows.

### 5.2 Direct segment uniforms

The host optical bake receives segment geometry and properties directly via uniform arrays (supporting up to 16 active segment layers per host):

- `uniform int u_segment_count;`: number of defined segments for the host.
- `uniform vec4 u_segment_geometry[16];`: packed $(r_{\mathrm{in}}/R_{\mathrm{eq}},\, r_{\mathrm{out}}/R_{\mathrm{eq}},\, \mathrm{opacity},\, \mathrm{row\_idx})$.
- `uniform vec4 u_segment_props[16];`: packed $(g_f,\, g_b,\, \mathrm{scatter},\, \mathrm{unlit\_factor})$.
- `uniform float u_segment_textured[16];`: boolean material family flag ($1.0$ for textured, $0.0$ for procedural).

### 5.3 Overlapping materials: homogeneously mixed slab

Coplanar layers are modeled as a **homogeneously mixed slab**. They are not an averaged single effective material, nor are they a vertically ordered stack of physically separated layers.

At each radial quadrature node $r$:

1. The shader iterates over all $M$ active segments ($i=0,\dots,M-1$).
2. If $r$ lies outside $[r_{i,\mathrm{in}}, r_{i,\mathrm{out}}]$, layer $i$ contributes $\tau_i=0$.
3. If $r$ is within bounds, the profile coordinate is $u=(r-r_{i,\mathrm{in}})/(r_{i,\mathrm{out}}-r_{i,\mathrm{in}})$. The shader samples row `geometry.w` of `u_ring_gradients`:
   $$
   \alpha_i=\operatorname{clamp}(\mathrm{tex.a}\times\mathrm{opacity}_i,\, 0,\, 1-2^{-23}),
   $$
   $$
   \tau_i=\operatorname{ring\_tau}(\alpha_i),
   \qquad
   C_{i,\mathrm{lin}}=\max(\mathrm{tex.rgb},0)^{2.2}.
   $$
4. The total slab optical depth is the sum of layer normal optical depths:
   $$
   \tau_{\mathrm{total}}=\sum_{i=1}^M \tau_i.
   $$
   If $\tau_{\mathrm{total}}\le 0$, the radial node is transparent and skipped.
5. In the inner azimuth quadrature, `ringshine_material_radiance` evaluates:
   - Slab transfer $(S_{\mathrm{lit/opp}}, S_{\mathrm{ms}})$ using the shared $\tau_{\mathrm{total}}$, viewing cosine $\mu_v$, and solar incidence $\mu_0$.
   - Lit-face opposition surge using $\tau_{\mathrm{total}}$ and phase cosine $\mu$.
   - For each layer with $\tau_i>0$, its own phase function $p_i$ and multiple-scattering weight $p_{\mathrm{ms},i}$ from its own $\alpha_i$ and properties.
   - Layers are summed with optical depth fraction weights:
     $$
     \sum_{i=1}^M C_{i,\mathrm{lin}}\left(\frac{\tau_i}{\tau_{\mathrm{total}}}\right)
     \left[S_{\mathrm{single},i}\,p_i + S_{\mathrm{ms}}\,p_{\mathrm{ms},i}\right].
     $$

Neither asymmetry nor material family is averaged before optical evaluation. Both Henyey–Greenstein and Cornette–Shanks layers coexist at the same radius and retain their distinct forward/backward lobes without mode switching or cancellation.

## 6. Oblate receiver geometry and the visible ring arc

### 6.1 Receiver position and normal

In local normalized coordinates, the host is

$$
x^2+y^2+\eta z^2=1.
$$

For geocentric latitude $\lambda$, its radius is

$$
\rho=\frac{q}{\sqrt{q^2c^2+s^2}}.
$$

Fold the receiver into the positive-$z$ hemisphere for geometric calculation and place its meridian on the local $x$ axis:

$$
P=(\rho c,0,\rho s),\qquad
Q(a)=(r\cos a,r\sin a,0).
$$

The normalized surface normal follows the spheroid gradient:

$$
N_p=\frac{(c,0,\eta s)}{n},
\qquad n=\sqrt{c^2+\eta^2s^2}.
$$

The sign of the original hemisphere is retained separately to determine which ring face is viewed and the scattering angle.

Define

$$
h=\rho s,\quad
d_0^2=(r-\rho c)^2+h^2,\quad
r_c=r\rho c.
$$

Then

$$
d^2=d_0^2+4r_c\sin^2(a/2).
$$

The half-angle form avoids subtracting nearly equal large terms for a ring element close to the surface meridian.

The two geometric cosines are

$$
\mu_v=\frac{h}{d},
\qquad
\mu_p=
\max\left(
0,\frac{rc\cos a-1/\rho}{nd}
\right).
$$

The numerator in the receiver term follows from $N_p\cdot(Q-P)$ and the spheroid surface equation. The shader writes it equivalently as

$$
rc-\frac1\rho-2rc\sin^2(a/2).
$$

Here $rc$ means $r\,c$; it is distinct from the auxiliary $r_c=r\rho c$ used in the distance expression.

### 6.2 Exact host horizon

The ring element lies above the tangent-plane horizon precisely when

$$
r\rho c\cos a>1.
$$

If $r\rho c\le1$, the entire radial circle is below the horizon and contributes zero.

Otherwise its visible interval is

$$
-a_{\max}<a<a_{\max},\qquad
a_{\max}=\arccos\left(\frac1{r\rho c}\right).
$$

For a convex spheroid, this outward-facing visibility condition also rules out an intervening intersection with the host. No per-node receiver-to-ring ray march is needed.

Two special cases are explicit:

- **Equator:** $s<10^{-7}$ returns zero. An infinitely thin coplanar ring has no projected area there.
- **Poles:** $c=0$ fails the horizon condition. The equatorial ring is below the polar surface horizon.

Flattening changes $\rho$, the normal, distances, and visible arcs together. It is not implemented as a brightness correction applied to spherical geometry.

## 7. The host's shadow on the ring

A geometrically visible ring element still contributes nothing if the host blocks its starlight. The bake computes this shadow analytically for a directional star and an oblate host.

Let

$$
C=\sqrt{1-b^2},\qquad A=C^2+\eta b^2.
$$

For $C>10^{-7}$, define

$$
z_{\mathrm{shadow}}
=\frac{\sqrt{(1-r^{-2})A}}{C}.
$$

If $z_{\mathrm{shadow}}\ge1$, there is no shadow interval on that radial circle. Otherwise the angular half-width is

$$
\delta=\arccos(z_{\mathrm{shadow}}).
$$

The shadowed interval is centered on the anti-solar direction:

$$
[\phi-\delta,\phi+\delta],
$$

with copies shifted by $2\pi$ as necessary.

### 7.1 Derivation from a ray–spheroid intersection

For a ring point $Q$, consider the ray $Q+tS$ toward the star. With the star's projected direction opposite the shadow center,

$$
S=(-C\cos\phi,-C\sin\phi,b).
$$

Substitution into the spheroid equation gives

$$
A t^2+2Bt+(r^2-1)=0,\qquad
B=-rC\cos(a-\phi).
$$

A forward intersection requires the point to be on the anti-solar side and the quadratic discriminant to be positive:

$$
B<0,\qquad B^2-A(r^2-1)>0.
$$

Solving its tangent boundary yields the expression for $\delta$. This directly incorporates host flattening.

### 7.2 Integrate only the surviving arcs

`ringshine_band` starts with the receiver-visible interval $[-a_{\max},a_{\max}]$. It subtracts the host shadow using copies centered at $\phi-2\pi$, $\phi$, and $\phi+2\pi$. A cursor walks the interval and sends each remaining illuminated subinterval to `ringshine_arc`.

This matters numerically: the shadow discontinuity becomes an integration boundary, not a binary mask that a few quadrature nodes might miss. Geometry and scattering are then evaluated together along each surviving arc.

“Exact lit arcs” does **not** mean a closed-form radiometric solution. The arc endpoints are analytic; the radiance integral over them is numerical.

### 7.3 Extended source integration and equinox continuity

The bake receives the stellar angular radius sine via `u_sun_radius` ($\sin r_\star$). When resolved (`ring_resolve_disk`: $\sin r_\star > 0$ and $(|b| < 2\sin r_\star \text{ or } \sin r_\star > 0.001)$), starlight is integrated as a uniform-radiance circular disk using 64 samples (4 Gauss nodes in projected radius squared $\times$ 16 symmetric azimuth nodes), implemented in [`common/ring_source.glsl`](../engine/glsl/common/ring_source.glsl).

For each sample $q \in [0, 63]$:
1. `ring_disk_sample` generates direction vector $S_q$, with signed elevation $b_q=S_{q,y}$ and azimuth shift $\Delta\phi_q=\operatorname{atan2}(-S_{q,z},-S_{q,x})$.
2. The quadrature weight $w_{\star,q}=\frac{W_{\mathrm{row}}}{16\sqrt{1-r^2}}$ integrates $\frac{d\Omega}{\pi\sin^2 r_\star}$.
3. `ringshine_band` evaluates the oblate host shadow and horizon-visible arcs for that sample direction $(b_q, \phi + \Delta\phi_q)$.
4. When a star crosses the ring plane ($|b| < \sin r_\star$), individual samples have both positive and negative elevations. Starlight illuminates both faces of the ring simultaneously, each with its sampled elevation and phase.

This removes the step discontinuity and blackout at equinox ($b=0$) present in a directional point-source model. Smaller stars away from the plane ($\sin r_\star \le 0.001$ and $|b| \ge 2\sin r_\star$) use the point limit ($N_\star=1$) at the disk center with $w_{\star,0}=1.0$.

For each individual sample ray, the host shadow on the ring retains a hard geometric edge. Integrating 64 directional samples across the disk turns those hard edges into a smooth penumbra on the ring. External eclipsers remain outside this integral, and direct-light caster penumbrae retain their analytic approximation.

## 8. Ring radiance: finite-optical-depth slab transfer

The ring is treated as an infinitesimally thin geometric sheet with a finite optical-depth slab response. Geometry supplies its projected area; optical depth supplies scattering and attenuation.

### 8.1 Optical depth and continuous conversion

`alpha` is vertical extinction opacity, never camera-path opacity. In [`common/ring_optics.glsl`](../engine/glsl/common/ring_optics.glsl), `ring_tau(alpha)` clamps $\alpha$ to $[0, 1 - 2^{-23}]$ (`RING_MAX_ALPHA = 0.9999999`) and converts it continuously to normal optical depth:

$$
\tau(\alpha)=
\begin{cases}
\alpha\left(1+\alpha\left(\frac12+\frac\alpha3\right)\right), & \alpha < 0.001,\\
-\ln(1-\alpha), & \alpha \ge 0.001.
\end{cases}
$$

The series expansion prevents floating-point cancellation for thin dust profiles in 32-bit arithmetic. On the CPU, [`ring_optics.py`](../engine/rendering/ring_optics.py) uses `-math.log1p(-alpha)`.

For total optical depth $\tau_{\mathrm{total}}$ at radius $r$, the slant depths under viewing cosine $\mu_v$ and solar incidence cosine $\mu_0=|b|$ are:

$$
v=\frac{\tau_{\mathrm{total}}}{\max(\mu_v,10^{-7})},
\qquad
l=\frac{\tau_{\mathrm{total}}}{\max(\mu_0,10^{-7})}.
$$

Zero opacity or $\mu_0\le 0$ produces zero response.

The receiver sees the illuminated ring face when

$$
\ell b\ge 0.
$$

This tests which side of the ring plane contains the star sample and the receiver. It does not test whether the planet's surface point itself is in daylight.

### 8.2 Illuminated-face single scattering and opposition

The illuminated-face slab factor is

$$
S_{\mathrm{lit}}
=
\frac{\mu_0}{\mu_v+\mu_0}
\left[1-e^{-(v+l)}\right].
$$

For small total slant depth $x=v+l$, `ring_absorbed(x)` evaluates

$$
1-e^{-x}\approx x\left(1-x\left(\frac12-\frac{x}{6}\right)\right)
\quad (x<0.001)
$$

to avoid cancellation. At high optical depth, the illuminated-face factor saturates at $\mu_0/(\mu_v+\mu_0)$.

Lit-face single scattering is multiplied by the authored opposition surge:

$$
\theta_{\mathrm{opp}}=\arccos(\operatorname{clamp}(-\mu,-1,1)),
\qquad
d_\tau=\operatorname{clamp}\left(\frac{\tau_{\mathrm{total}}}{1.5},0,1\right),
$$

$$
\Omega(\mu,\tau_{\mathrm{total}})=
\left(1+\frac{0.8 d_\tau}{1+\theta_{\mathrm{opp}}/0.07}\right)
\left(1+0.3 d_\tau\,e^{-\theta_{\mathrm{opp}}/0.006}\right).
$$

This opposition term now applies consistently across visible rings, host ringshine maps, and CPU lighting.

### 8.3 Opposite-face single scattering

For a receiver on the opposite side of the ring from the star, the code uses

$$
\Delta=|l-v|,
$$

$$
S_{\mathrm{opp}}
=
v\,e^{-\min(v,l)}
\frac{1-e^{-\Delta}}{\Delta}.
$$

At $\Delta=0$, the ratio has limit one:

$$
S_{\mathrm{opp}}=v e^{-v}\quad\text{when }v=l.
$$

The GPU evaluates the ratio stably as:

$$
\frac{1-e^{-\Delta}}{\Delta}
\approx 1-\Delta\left(\frac12-\frac\Delta6\right)
\quad(\Delta<0.001).
$$

The CPU equivalent uses `-math.expm1(-delta)/delta` with an explicit equal-depth limit. Opposite-face single scattering is scaled by the layer's `unlit_factor` (`max(0.0, props.a)`).

### 8.4 Why layers and optical depth must not be averaged before transfer

Opacity becomes optical depth through a logarithm, then enters exponential attenuation and nonlinear phase-function weighting:

$$
R(\operatorname{average}(\alpha))
\ne \operatorname{average}(R(\alpha)).
$$

The model evaluates optical depth and phase response at each radial quadrature node independently. Furthermore, across overlapping coplanar layers, each layer evaluates its own phase response with its own $\alpha_i$ and properties, weighted by $\tau_i / \tau_{\mathrm{total}}$. Neither radial bands nor overlapping layers are reduced to an averaged effective material.

## 9. Phase functions and the multiple-scattering approximation

### 9.1 Scattering-angle convention

The phase cosine compares the incoming photon travel direction with the outgoing direction from the ring to the surface:

$$
\chi=\mu=(-S)\cdot\frac{P-Q}{d}=-\operatorname{dot}(L,V).
$$

Thus $\mu=1$ is forward scattering and $\mu=-1$ is backscattering. In the folded geometry, the shader evaluates

$$
\mu=
\frac{
(\rho c-r\cos a)C\cos\phi
-r\sin a\,C\sin\phi
+\sigma h|b|
}{d},
$$

where $\sigma=-1$ on the illuminated face and $+1$ on the opposite face, clamped to $[-1,1]$.

Angle conventions matter when comparing formulas: [PBRT's Phase Functions](https://www.pbr-book.org/4ed/Volume_Scattering/Phase_Functions) uses two directions pointing away from the interaction and therefore writes the HG denominator with a different sign. The formula below uses this project's photon-travel convention.

### 9.2 The two analytic lobes

Henyey–Greenstein (HG) is implemented as:

$$
p_{\mathrm{HG}}(g,\mu)
=
\frac{1-g^2}
{4\pi d_g^{3/2}},
\qquad
d_g=(1-|g|)^2+2|g|(1-\operatorname{sgn}(g)\mu).
$$

Cornette–Shanks (CS) is implemented as:

$$
p_{\mathrm{CS}}(g,\mu)
=
p_{\mathrm{HG}}(g,\mu)
\frac{3(1+\mu^2)}{2(2+g^2)}.
$$

Both lobes integrate to exactly 1 over solid angle ($4\pi$). Both $g_f$ (`asymmetry`) and $g_b$ (`backscatter`) are shape parameters clamped to $[-0.99, 0.99]$. CS's $g$ is a shape parameter, not strictly its mean scattering cosine. The second parameter may be positive, allowing two forward lobes.

The combined phase response for a layer is:

$$
p_i = w_f\,p_f + (1-w_f)\,p_b.
$$

The two lobe weights sum to one. `backscatter` is the second lobe's shape parameter, not its blend weight.

### 9.3 Textured and procedural material families

For textured layers (`u_segment_textured[i] > 0.5`):

$$
t_{\mathrm{chunks}}=\operatorname{clamp}\left(\frac{\alpha_i-0.1}{0.5},0,1\right),
$$

$$
w_f=0.95-0.45 t_{\mathrm{chunks}},
\qquad
w_{\mathrm{ms}}=t_{\mathrm{chunks}}.
$$

Both lobes use Henyey–Greenstein. Low-opacity material behaves as forward-scattering dust, while denser material includes more large-particle/backward response and multiple scattering.

For procedural layers (`u_segment_textured[i] <= 0.5`):

$$
t_{\mathrm{balance}}=\operatorname{clamp}(\mathrm{scatter},0,1),
$$

$$
w_f=0.1+0.8 t_{\mathrm{balance}},
\qquad
w_{\mathrm{ms}}=\operatorname{clamp}(1-0.7 t_{\mathrm{balance}},0.1,1).
$$

Both lobes use Cornette–Shanks. In the textured branch, the `scatter` balance parameter is unused.

### 9.4 Multiple scattering and homogeneous mixture

The isotropic multiple-scattering approximation uses albedo $w_0=0.92$ and $\gamma=\sqrt{1-w_0}=\sqrt{0.08}$:

$$
H(\mu)=\frac{1+2\mu}{1+2\mu\gamma}.
$$

The slab multiple-scattering transfer factor is:

$$
S_{\mathrm{ms}}
=
\frac{0.92}{4\pi}
\frac{\mu_0}{\mu_v+\mu_0}
\max[0,H(\mu_v)H(\mu_0)-1]
\left[1-e^{-(v+l)}\right].
$$

At each radial node, the total radiance toward the receiver is accumulated over active layers:

$$
L_r(\mu_v,\mu_0,\mu)=
\sum_{i=1}^M
C_{i,\mathrm{lin}}
\left(\frac{\tau_i}{\tau_{\mathrm{total}}}\right)
\left[
S_{\mathrm{single},i}\,p_i + S_{\mathrm{ms}}\,w_{\mathrm{ms},i}
\right],
$$

where:

$$
S_{\mathrm{single},i}=
\begin{cases}
S_{\mathrm{lit}}\,\Omega(\mu,\tau_{\mathrm{total}}), & \ell b \ge 0,\\
S_{\mathrm{opp}}\,\max(0,\mathrm{unlit\_factor}_i), & \ell b < 0.
\end{cases}
$$

All paths (visible rings, host ringshine, and CPU moon illumination) evaluate this identical optical formulation. Single scattering uses the layer's own phase function and opacity, while both multiple scattering and lit-face opposition depend on total slab depth $\tau_{\mathrm{total}}$ and are weighted by each layer's optical depth fraction $\tau_i/\tau_{\mathrm{total}}$.

## 10. Numerical integration

### 10.1 Azimuth: 16-point Gaussian quadrature with a geometric warp

A naive small set of uniformly distributed azimuth samples can miss the narrow contribution peak near the closest ring element, especially when the inner ring nearly touches the host.

For every surviving arc $[a_0,a_1]$, the implementation defines

$$
k=\sqrt{\frac{d_0^2}{\max(r\rho c,10^{-12})}},
\qquad
a=k\tan\beta.
$$

Its transformed limits and Jacobian are

$$
\beta_0=\arctan(a_0/k),\quad
\beta_1=\arctan(a_1/k),\qquad
\frac{da}{d\beta}=k(1+\tan^2\beta).
$$

Near $a=0$,

$$
d^2\approx d_0^2+r\rho c\,a^2.
$$

The scale $k$ therefore follows the angular width of the near-contact geometry. Uniform Gaussian nodes in $\beta$ concentrate in $a$ where the closest source region needs resolution.

Using the fixed 16-point Gauss–Legendre nodes $x_i$ and weights $w_i$ in the shader,

$$
\beta_i=
\frac{\beta_0+\beta_1}{2}
+\frac{\beta_1-\beta_0}{2}x_i,
$$

$$
I_{\mathrm{arc}}\approx
\frac{\beta_1-\beta_0}{2}
\sum_{i=1}^{16}
w_i
R_i\frac{\mu_{p,i}\mu_{v,i}}{d_i^2}
\,r\,k(1+\tan^2\beta_i).
$$

The $r$ factor is already included here. It must not be applied again in the radial weight.

The method has a fixed node count, not an adaptive error estimator. Its warp adapts to geometry, not explicitly to the width of the scattering lobes.

### 10.2 Radius: logarithmic gap spacing and four nodes per band

The usable inner radius is

$$
r_{\mathrm{first}}=\max(r_i,1.000001).
$$

Ring material at or inside the equatorial host radius is omitted. If $r_o\le r_{\mathrm{first}}$, the tile is zero. The material coordinate still spans the original $[r_i,r_o]$ domain; clipping geometry does not shift the texture.

Set

$$
g_0=r_{\mathrm{first}}-1,\qquad
L_g=\ln\left(\frac{r_o-1}{g_0}\right).
$$

For $t\in[0,1]$,

$$
g(t)=g_0e^{tL_g},\quad r(t)=1+g(t),
\qquad
\frac{dr}{dt}=g(t)L_g.
$$

The interval is divided into $B$ bands, with $B$ clamped to 4–1024. Four Gauss–Legendre nodes and weights on $[0,1]$ are used in each band:

| Node | $T_s$ | $W_s$ |
| --- | --- | --- |
| 0 | 0.0694318442 | 0.1739274226 |
| 1 | 0.3300094782 | 0.3260725774 |
| 2 | 0.6699905218 | 0.3260725774 |
| 3 | 0.9305681558 | 0.1739274226 |

For band $m$ and node $s$,

$$
t_{m,s}=\frac{m+T_s}{B},
\qquad
w_{r,m,s}=\frac{g(t_{m,s})L_gW_s}{B}.
$$

At each radial node $r(t_{m,s})$, the shader evaluates active segments from `u_segment_geometry`:

1. For each segment $i \in [0,\dots,M-1]$, if $r \in [r_{i,\mathrm{in}}, r_{i,\mathrm{out}}]$, it samples row `geometry.w` of `u_ring_gradients` at $u=(r-r_{i,\mathrm{in}})/(r_{i,\mathrm{out}}-r_{i,\mathrm{in}})$, extracting $\tau_i=\operatorname{ring\_tau}(\alpha_i)$ and linear RGB color.
2. It sums $\tau_{\mathrm{total}}=\sum_i \tau_i$. If $\tau_{\mathrm{total}}\le 0$, the node is transparent and skipped.
3. If `ring_resolve_disk(sun, u_sun_radius)` is true (64 samples), it iterates over disk samples; otherwise it uses the single center sample ($N_\star=1$):
   $$
   I_{\mathrm{node}}(r_{m,s})=\sum_{q=1}^{N_\star} w_{\star,q}\,I_{\mathrm{band}}(r_{m,s},S_q).
   $$
4. It weights the radial contribution:
   $$
   K\approx
   \sum_{m=0}^{B-1}\sum_{s=0}^{3}
   I_{\mathrm{node}}(r_{m,s})
   \,w_{r,m,s}.
   $$

Logarithmic spacing in **distance from the host surface** puts radial effort near contact. Far from the host, it approaches log-radius spacing.

Bands are not aligned to every material discontinuity or ringlet. Four transfer evaluations per band improve nonlinear sampling but do not guarantee convergence for arbitrary fine profiles.

### 10.3 Schematic bake

~~~text
for each texel of a dirty host/star tile:
    decode geocentric sine latitude (slat) and anti-solar azimuth (phi)
    sun = clamp(u_sun_elevation, -1.0, 1.0)
    sources = ring_resolve_disk(sun, u_sun_radius) ? 64 : 1
    total = RGB(0)

    for each logarithmic radial band m in [0, B-1]:
        for each Gaussian node s in [0, 3]:
            compute radius r and profile coordinate u_i for each segment
            sample u_ring_gradients segment rows: layer_alpha, layer_tau, layer_color
            material_tau = sum(layer_tau)
            if material_tau <= 0: continue

            node_response = RGB(0)
            for each source sample q in [0, sources-1]:
                sample = sources == 1 ? center : ring_disk_sample(center, pole, radius, q)
                sample_phi = phi + atan(-sample.z, -sample.x)
                find ring circle's horizon-visible interval [-limit, limit]
                find host shadow interval for sample.y and sample_phi

                for each surviving illuminated arc:
                    integrate ringshine_material_radiance × mu_p × mu_v / d² × r
                    using 16 geometrically warped Gaussian nodes

                node_response += sample.w × arc_integral

            total += node_response × dr_weight

    write RGB = pi × total
~~~

## 11. The cached map and its angular parameterization

### 11.1 Atlas allocation

`RingshineMap` allocates one RGBA float32 texture:

- **Tile:** 128 × 65 texels.
- **Columns:** 16 independent stars.
- **Rows:** 16 independent ring hosts.
- **Whole atlas:** 2048 × 1040 texels.
- **Raw texel storage:** 34,078,720 bytes, or 32.5 MiB, excluding driver overhead.

Tile $(k,j)$ has framebuffer viewport

~~~text
x = j × 128
y = k × 65
width = 128
height = 65
~~~

This atlas is distinct from the radial material/property atlases. Host row $k$ and star column $j$ must have consistent identities between the CPU update and all consumers.

RGB is the transfer value $M$. Alpha is not used by the surface lookup; normal bake output sets it to one, while whole-tile early exits write zero RGBA.

Filtering is linear, wrapping is disabled, and sampling uses mip level zero.

### 11.2 Closed angular domain at texel centers

For integer local texel coordinates $(i,j)$,

$$
x=2\frac{i}{127}-1,\qquad
y=2\frac{j}{64}-1.
$$

The bake decodes

$$
\phi=\pi\,\operatorname{sgn}(x)|x|^{3/2},
\qquad
\ell=\operatorname{sgn}(y)|y|^{3/2}.
$$

The power warp places more samples near the equator and anti-solar meridian, where near-contact transport and the host shadow can change rapidly.

Important grid details:

- The first and last rows represent the poles exactly.
- The middle row, $j=32$, represents the equator exactly.
- Both azimuth seam endpoints, $-\pi$ and $+\pi$, are represented.
- Width 128 is even, so there is no column exactly at $\phi=0$.

This is a closed-domain sampling scheme. Treating texel centers as ordinary unadjusted $i/128$ coordinates would shift the endpoints and lose the intended pole/equator placement.

## 12. Surface lookup and composition

### 12.1 Select the host and derive the lookup direction

The sphere and solid-terrain shaders loop over registered ring hosts. They first reject distant or degenerate candidates, then identify a host receiver using the caster data:

- With a matching host caster, the fragment must lie within 1.5 host radii.
- If the caster is unavailable, the fallback is distance inside the ring's inner radius.

This is a geometric host-selection heuristic, not a separate receiver-body ID stored in the map.

For a matched host, the radial direction $P_{\mathrm{dir}}$ supplies

$$
\ell=P_{\mathrm{dir}}\cdot N_r.
$$

Project $P_{\mathrm{dir}}$ and the anti-stellar direction $-L$ into the ring plane, normalize them, and call the results $P_{\mathrm{eq}}$ and $A_{\mathrm{eq}}$. Then

$$
\phi=
\operatorname{atan2}\left[
N_r\cdot(A_{\mathrm{eq}}\times P_{\mathrm{eq}}),
P_{\mathrm{eq}}\cdot A_{\mathrm{eq}}
\right].
$$

This measures surface longitude relative to the current star rather than to a fixed world axis. It lets a cached tile rotate with the star without a rebake.

The bake uses the star direction at the host center. The surface lookup uses the fragment-to-star direction. They agree in the distant-star limit assumed by the transport model; finite-distance parallax across a very large host/ring system is not fully modeled.

### 12.2 Invert the angular warp

The inverse coordinates are

$$
u=\frac12+\frac12\operatorname{sgn}(\phi/\pi)
|\phi/\pi|^{2/3},
$$

$$
v=\frac12+\frac12\operatorname{sgn}(\ell)|\ell|^{2/3}.
$$

The current caller packs `host_uv = (u, (k+v)/16)`. `ringshine_sample` extracts local $v$ again as `host_uv.y * 16 - k`, then computes

$$
u_t=\frac{0.5+\operatorname{clamp}(u,0,1)(128-1)}{128},
$$

$$
v_t=\frac{0.5+\operatorname{clamp}(v,0,1)(65-1)}{65}.
$$

The atlas coordinate is

$$
UV_{\mathrm{atlas}}=
\left(\frac{j+u_t}{16},\frac{k+v_t}{16}\right).
$$

These half-texel margins restrict bilinear filtering to the selected tile. Sampling directly on tile borders could mix another star or host into the result.

### 12.3 Apply the current stellar and surface factors

For each star, the shaders interpolate its equatorial and polar color/luminosity using the absolute cosine between the fragment-to-star direction and the star's pole. They multiply the map by that RGB, by runtime falloff, and by $1/\pi$.

The summed ringshine is multiplied by the receiver's surface albedo and added to the other illumination. Exposure is handled downstream in the relevant surface path.

The host's horizon and smooth oblate incidence cosine are already baked. **There is no additional ringshine dot product against the fragment normal.** Surface normal maps, terrain slopes, and local relief therefore do not redirect or occlude this integrated source.

Ringshine also retains a diffuse receiver approximation for Hapke materials, as documented in [surface_materials.md](surface_materials.md). The map does not preserve incident-direction information with which to evaluate a directional surface BRDF.

The surface's direct-sun eclipse/shadow multiplier is not applied to this host ringshine term. The bake already handles the host's shadow on the source ring, but it does not include other bodies obscuring the ring or its illumination.

## 13. Cache identity, invalidation, and GPU state

### 13.1 One key per host/star tile

For each active host and each of at most 16 stars, `RingshineMap.update` builds

~~~text
tile = (host_index, star_index)

key = (
    gradient.profile_revision,
    (inner_radius / host_radius,
     outer_radius / host_radius,
     host_enable,
     effective_flattening),
    clamped_band_count,
    quantized_signed_solar_sine_elevation,
    quantized_stellar_angular_radius_sine
)
~~~

The signed elevation is calculated using float64 world-space positions:

$$
b=
\frac{(X_\star-X_{\mathrm{host}})\cdot N_r}
{\lVert X_\star-X_{\mathrm{host}}\rVert},
$$

then rounded to five decimal places and clamped to $[-1,1]$.

Using world-space positions avoids camera-origin rebasing perturbing this key. The ring normal is expected to be a valid unit pole.

The maximum rounding error is $5\times10^{-6}$ in **sine elevation**, not a fixed angular error.

The stellar angular radius sine is computed as:

$$
\sin r_\star=\min\left(0.999,\max\left(0.0,\frac{R_\star}{\lVert X_\star-X_{\mathrm{host}}\rVert}\right)\right),
$$

then formatted to six significant digits (`.6g`). This relative quantization accurately captures small stellar disks near equinox while avoiding spurious rebakes from orbital floating-point noise.

### 13.2 What triggers a rebake?

| Change | Effect |
| --- | --- |
| Segment profile, geometry, or optical property edit | Global `profile_revision` changes; active tiles become dirty |
| Normalized inner/outer radius | Affected host's tiles become dirty |
| Host enable state | Affected host's tiles become dirty |
| Effective host flattening | Affected host's tiles become dirty |
| Band count | Active tiles become dirty |
| Star elevation crossing a quantization step ($5\times 10^{-6}$) | That host/star tile becomes dirty |
| Stellar angular radius change ($\sin r_\star$) | That host/star tile becomes dirty |
| Camera translation, orientation, or rebasing | Reuses the map |
| Star azimuth at unchanged elevation | Reuses the map; lookup coordinates rotate |
| Stellar color or luminosity | Reuses the map; applied during lookup |
| Star distance with unchanged elevation and $\sin r_\star$ | Reuses the map; falloff changes at lookup |
| Host spin about its unchanged ring pole | Reuses the map; surface coordinates move through it |
| Host/ring rescaling with unchanged normalized geometry | Reuses the map, subject to exact floating-point key equality |

With the oblateness option disabled, the effective flattening is zero. A change to actual body flattening then has no effect on the bake's geometry key.

The material revision is global, not per host. Even a localized edit can invalidate every active host/star tile. Geometry tuples are compared directly; solar elevation and stellar radius are the explicitly quantized components.

Inactive tile keys are removed from the cache after an update. Their texture contents need not be cleared because active counts/indices determine what can be sampled; reactivated tiles without a retained key are rebaked.

### 13.3 Drawing dirty tiles

Dirty tiles are rendered into their own viewports through a framebuffer scope with `enable_only=0`. This disables blend/depth effects during the overwrite, preventing the caller's draw state from accumulating or rejecting bake results.

The bake binds `u_ring_gradients` to texture unit 0. The obsolete property texture (`u_ring_props`) is removed. The shader receives segment geometry (`u_segment_geometry`), phase/optical properties (`u_segment_props`), textured flags (`u_segment_textured`), active segment count (`u_segment_count`), stellar angular radius (`u_sun_radius`), host parameters (`u_host_params`), solar elevation (`u_sun_elevation`), and band count via uniform writes. After the update, `app.py` restores the scene viewport and HDR resolve framebuffer.

`last_update_count` reports the number of tiles actually drawn by the most recent `update` call. It is useful for detecting unnecessary rebakes. The GPU timing region in `app.py` is named `gpu_ringshine_map`.

## 14. Cost, controls, and expected behavior

The default settings are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `ringshine_enabled` | `True` | Enable host ringshine contribution and host-map updates |
| `ringshine_oblate_enabled` | `True` | Include host flattening in map geometry |
| `ringshine_band_count` | `10` | Radial integration bands, allowed range 4–1024 |

The band count controls **radial** quadrature only. It does not increase tile dimensions or the 16-node azimuth rule.

A tile has 8,320 texels. For point stars ($N_\star=1$), the basic quadrature workload is:

$$
8320\times B\times 4\times 16 = 532{,}480B
$$

node evaluations per tile. When the star's finite disk is resolved at equinox or for large sources ($N_\star=64$), the workload is:

$$
8320\times B\times 4\times 64\times 16 \approx 34{,}078{,}720B
$$

node evaluations per tile before early-outs. Split lit arcs can add work; transparent nodes, horizon rejection, equatorial/polar rows, and zero illumination remove work. This is an operation-count estimate, not a measured timing.

Steady-state shading uses one bilinearly filtered texture sample per applicable host/star pair, plus coordinate and stellar-factor calculations. The expensive integration is paid when keys change. Continuously varying stellar elevation can still force rebakes at successive quantization steps.

Increasing $B$ helps radial material and near-contact convergence. It does not fix a narrow angular scattering lobe, a map that is too coarse over latitude/longitude, or a missing physical effect.

Typical qualitative behavior follows directly from the model:

- Ringshine can illuminate the planet's night side because the ring is an extended source.
- The host shadow removes some of that source, so the anti-solar pattern is not a uniform glow.
- The same-side and opposite-side hemispheres have different optical responses.
- Finite stellar disk quadrature removes the sharp equinox blackout ($b=0$); both faces of the ring are illuminated continuously across equinox.
- Multiple overlapping segments preserve distinct forward/backward lobes and material colors via homogeneous mixing without averaging artifacts.
- The exact equator and poles have zero geometric contribution in this thin-ring surface model.
- A gap changes illumination through its location, visibility, and material response, not merely its missing area.
- Flattening changes where the ring is visible as well as how much illumination arrives.

## 15. Worked geometry example

Choose a host flattening $f=0.1$, receiver latitude $\lambda=30^\circ$, ring circle $r=2$, and stellar sine elevation $b=0.2$. All lengths are in host equatorial radii.

The geometry gives

$$
q=0.9,\quad s=0.5,\quad c=\sqrt{3}/2,\quad
\rho\approx0.971909.
$$

The receiver can see the interval

$$
-a_{\max}<a<a_{\max},
\qquad a_{\max}\approx53.555946^\circ.
$$

The host's stellar shadow on this circle has half-width

$$
\delta\approx27.374463^\circ.
$$

For a receiver on the anti-solar meridian, $\phi=0$, the illuminated visible arcs are therefore approximately

$$
[-53.555946^\circ,-27.374463^\circ]
\quad\text{and}\quad
[27.374463^\circ,53.555946^\circ].
$$

At the closest ring element, $a=0$,

$$
d\approx1.256112,\quad
\mu_v\approx0.386872,\quad
\mu_p\approx0.526356.
$$

That closest element has favorable geometric weight but lies inside the host shadow in this example, so it contributes nothing. This illustrates why integrating only a form factor and multiplying by a coarse average shadow fraction would lose important structure.

## 16. Assumptions and limits

| Model choice | Consequence |
| --- | --- |
| Axisymmetric radial material profiles | No azimuthal spokes, localized arcs, or longitude-dependent texture transport |
| Infinitesimally thin planar ring | Exactly zero equatorial projected area; no finite-thickness illumination path |
| Ring and oblate host share a pole | Tilted rings relative to the host's figure are not represented by this derivation |
| Smooth convex reference spheroid | No local terrain horizon, height-dependent transfer, or normal-map response |
| Extended circular stellar disk quadrature | 64 samples when crossing the ring plane or $\sin r_\star > 0.001$; point limit otherwise. Direct-light caster penumbrae retain analytic approximation |
| Host-only analytic occlusion in the bake | External eclipsers and other ring systems are not integrated into this source visibility |
| Homogeneously mixed slab for overlapping segments | Coplanar layers evaluate individual phase responses with $\tau_i/\tau_{\mathrm{total}}$ weights; no property atlas |
| Authored phase families, opposition surge, and multiple scattering | Evaluated consistently across all paths using shared ring optics contract; not a measured spectral solution |
| Fixed angular quadrature and finite texture resolution | Increasing radial bands alone cannot remove every integration/interpolation error |
| 16 hosts and 16 stellar columns | Capacity is fixed across CPU allocation, viewport layout, and lookup conventions |
| Diffuse surface receiver | No directional Hapke/specular ringshine response |

The zero-thickness assumption can make the near-equatorial limit delicate when rings approach the surface: values just off the plane may change sharply while the exact equator remains zero. The radial and angular warps improve resolution there but do not change that geometric model.

The texture's 128 × 65 dimensions are hardcoded in the bake's angular decoding as well as defined on the CPU. Changing `RINGSHINE_TILE_SIZE` alone would not correctly change the system. The 16 × 16 atlas layout is also assumed by lookup code and surface callers.

## 17. Validation and debugging notes

The shared optical contract and ringshine implementation are verified by [`scripts/test_ring_optics.py`](../scripts/test_ring_optics.py):

- **GPU/CPU parity:** tests 144 optical combinations (extinction depths, viewing cosines, solar elevations, and material families) between GLSL and Numba CPU functions, asserting agreement within $4\times 10^{-5}$ relative tolerance.
- **Normalization:** both Henyey–Greenstein and Cornette–Shanks phase families integrate to exactly 1 over solid angle across extreme shape parameters ($\pm 0.99$).
- **Equinox continuity:** finite stellar disk quadrature crossing the ring plane agrees with dense numerical quadrature within 0.81%, eliminating edge-on step discontinuities.
- **Mixed slab CPU evaluation:** compiled Numba moon-lighting entry point (`ring_mixed_radiance`) preserves distinct forward/backward lobes under distant-body area quadrature.
- **Oblate geometry:** oblate horizon visibility and shadow half-width agree with direct spheroid-gradient normals and forward quadratic intersection tests.
- **Atlas assembly and cache invalidation:** atlas rebuild retains segment lobes without property averaging, and cache keys correctly invalidate upon stellar radius or profile revision updates.

For implementation work, the following checks distinguish different failure classes:

| Symptom | Useful check |
| --- | --- |
| Nearly constant glow instead of an anti-solar pattern | Verify $\phi$, the anti-stellar projection, and per-star tile selection |
| Too much or too little light by a nearly constant factor | Trace `M = pi * K` and the consumer's `1 / pi`; check for duplicated opacity or solar-cosine factors |
| Cross-contamination between hosts/stars | Verify tile indices and half-texel lookup margins |
| Stale appearance after a material edit | Confirm atlas rebuild increments `profile_revision` and uploads segment arrays |
| Equinox flash or blackout | Verify `u_sun_radius`, `ring_resolve_disk`, and 64-sample disk quadrature |
| Expensive rebakes while moving only the camera | Inspect `last_update_count` and world-space elevation inputs |
| Sharp-profile convergence changes with bands | Compare several radial band counts at fixed geometry and exposure |
| Near-contact error unchanged by more bands | Inspect the fixed azimuth quadrature and surface-map resolution |
| Equator/pole discrepancy | Distinguish exact model zeros from bilinear values nearby; verify closed-domain texel centers |
| Opposite-face NaNs or discontinuity | Check the coincident-depth limit of $(1-e^{-\Delta})/\Delta$ |
| Oblateness comparison looks like a simple gain | Verify that $\rho$, the surface normal, horizon, and shadow all use the same flattening |

Useful invariants for future validation include: transparent profiles yield zero; mirrored receiver latitude and star elevation preserve the response; each stellar contribution is additive after lookup; a pure azimuth change at fixed elevation reuses a tile; changing material revision invalidates all active tiles; and changing stellar angular radius invalidates the tile.

The source-of-truth route for future changes is:

~~~text
segment profiles & uniform arrays
    → RingshineMap.update cache key (includes sin_radius)
    → ringshine_map.frag radial quadrature
    → ring_resolve_disk (1 or 64 samples)
    → ring_disk_sample quadrature rays
    → ringshine_band oblate horizon & shadow
    → ringshine_arc warped azimuth quadrature
    → ringshine_material_radiance (shared ring_optics.glsl)
    → ringshine_sample tile lookup
    → surface albedo × stellar factors × M / pi
~~~
