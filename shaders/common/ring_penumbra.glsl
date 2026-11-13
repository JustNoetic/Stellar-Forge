// Per-frame ring penumbra table: one layer per (ring, star) slot, built at that star's incidence
// on the ring plane (constant across a view up to stellar parallax). R32F, RING_PENUMBRA_WIDTH
// columns, RING_PENUMBRA_ROWS+1 rows:
//   row 0: the raw ring extinction's prefix integral at this incidence, at the ring LUT's 4097
//          texel boundaries (columns 0..4096), so a box mean is an exact difference;
//   row r+1: the running integral along u of the extinction filtered by the three-box star-disk
//          kernel of half-width RING_PENUMBRA_H0 * 2^(r / RING_PENUMBRA_STEPS) (span units),
//          sampled over u in [-pad, 1 + pad] so the kernel's reach past the ring edges is kept.
// Point lookups apply the three-box kernel exactly to row 0 (six gathers). Swept boxes (the ring
// radii a ray piece covers) difference a kernel row, which keeps them flat boxes. Every read is a
// textureGather with fp32 weights: the hardware filter's 8-bit weights, magnified by deep ring
// shadow where little light is left, made blocky errors.
const int RING_PENUMBRA_WIDTH = 6144;
const int RING_PENUMBRA_ROWS = 56;
const float RING_PENUMBRA_STEPS = 4.0;
const float RING_PENUMBRA_H0 = 0.75/4096.0;

float ring_penumbra_row_width(float row) {
    return RING_PENUMBRA_H0*exp2(row/RING_PENUMBRA_STEPS);
}

float ring_penumbra_row_pad(float row) {
    return ring_penumbra_row_width(row)+2.0/4096.0;
}

// Ring coordinate of column x in kernel row `row`, and the column spacing.
float ring_penumbra_column_u(float row, float x) {
    float pad = ring_penumbra_row_pad(row);
    return -pad+x*(1.0+2.0*pad)/float(RING_PENUMBRA_WIDTH-1);
}

float ring_penumbra_column_spacing(float row) {
    return (1.0+2.0*ring_penumbra_row_pad(row))/float(RING_PENUMBRA_WIDTH-1);
}

#ifndef RING_PENUMBRA_BUILD
// Linear interpolation between columns i0 and i0+1 of texture row y, with fp32 weight f.
float ring_penumbra_gather(sampler2DArray table, float slot, float y, float i0, float f) {
    vec4 g = textureGather(table, vec3((i0+1.0)/float(RING_PENUMBRA_WIDTH), (y+1.0)/float(RING_PENUMBRA_ROWS+1), slot), 0);
    return mix(g.w, g.z, f);
}

// Raw extinction integral over [0, u] (span units), constant outside the ring.
float ring_penumbra_prefix(sampler2DArray table, float slot, float u) {
    float x = clamp(u, 0.0, 1.0)*4096.0;
    float i0 = min(floor(x), 4095.0);
    return ring_penumbra_gather(table, slot, 0.0, i0, x-i0);
}

// Mean raw extinction over [u-a, u+a], zero outside the ring (as ring_box_extinction).
float ring_penumbra_box(sampler2DArray table, float slot, float u, float a) {
    return max(ring_penumbra_prefix(table, slot, u+a)-ring_penumbra_prefix(table, slot, u-a), 0.0)/max(2.0*a, 1e-20);
}

// Kernel-filtered extinction at u for kernel half-width h: ring_shadow_filter's three boxes, exact,
// or the continuous kernel for wide penumbrae (see ring_penumbra_swept).
float ring_penumbra_extinction(sampler2DArray table, float slot, float u, float h);

// Integral of kernel row `row` (filtered extinction) over [-pad, u].
float ring_penumbra_row_integral(sampler2DArray table, float slot, float row, float u) {
    float pad = ring_penumbra_row_pad(row);
    float x = clamp((u+pad)/(1.0+2.0*pad)*float(RING_PENUMBRA_WIDTH-1), 0.0, float(RING_PENUMBRA_WIDTH-1));
    float i0 = min(floor(x), float(RING_PENUMBRA_WIDTH-2));
    return ring_penumbra_gather(table, slot, row+1.0, i0, x-i0);
}

float ring_penumbra_row_box(sampler2DArray table, float slot, float row, float u, float s) {
    return (ring_penumbra_row_integral(table, slot, row, u+s)-ring_penumbra_row_integral(table, slot, row, u-s))/(2.0*s);
}

// Kernel row level (fractional row index) for half-width h, clamped to the table, and the dilution
// that keeps the total for widths past the top row.
float ring_penumbra_level(float h, out float scale) {
    float level = max(log2(max(h, 1e-20)/RING_PENUMBRA_H0)*RING_PENUMBRA_STEPS, 0.0);
    float top = float(RING_PENUMBRA_ROWS-1);
    scale = level > top ? exp2((top-level)/RING_PENUMBRA_STEPS) : 1.0;
    return min(level, top);
}

// Box of half-width s over the kernel rows of half-width h: interpolated across the two rows
// bracketing h, or the nearest row once the box dominates (s >= 3h).
float ring_penumbra_rows(sampler2DArray table, float slot, float u, float h, float s) {
    float scale;
    float level = ring_penumbra_level(h, scale), top = float(RING_PENUMBRA_ROWS-1);
    if (s >= 3.0*h) return ring_penumbra_row_box(table, slot, min(floor(level+0.5), top), u, s)*scale;
    float row = floor(level), t = level-row;
    float value = ring_penumbra_row_box(table, slot, row, u, s);
    if (t > 0.0) value = mix(value, ring_penumbra_row_box(table, slot, min(row+1.0, top), u, s), t);
    return value*scale;
}

// Wide penumbrae use a continuous kernel: a box of RING_PENUMBRA_SMOOTH_BOX*h over the three-box
// kernel at RING_PENUMBRA_SMOOTH_WIDTH*h, which has the three-box kernel's variance and is as close to
// a uniform star disk's chord profile. The three boxes' hard edges echo each narrow gap as a sharp
// line when the gap is far narrower than the penumbra; in deep shadow (an opaque ring at low
// incidence) those small steps are much of the light that is left. Blended in over h in
// RING_PENUMBRA_SMOOTH_RANGE (span units), where the rows resolve it.
const float RING_PENUMBRA_SMOOTH_BOX = 0.2;
const float RING_PENUMBRA_SMOOTH_WIDTH = 0.972;
const vec2 RING_PENUMBRA_SMOOTH_RANGE = vec2(6.0, 12.0)/4096.0;

float ring_penumbra_smooth_blend(float h) {
    return clamp((h-RING_PENUMBRA_SMOOTH_RANGE.x)/(RING_PENUMBRA_SMOOTH_RANGE.y-RING_PENUMBRA_SMOOTH_RANGE.x), 0.0, 1.0);
}

float ring_penumbra_smooth(sampler2DArray table, float slot, float u, float h, float s) {
    float b = RING_PENUMBRA_SMOOTH_BOX*h;
    return ring_penumbra_rows(table, slot, u, RING_PENUMBRA_SMOOTH_WIDTH*h, sqrt(s*s+b*b));
}

// Flat box of half-width s convolved with the kernel of half-width h. Narrow penumbrae: short sweeps
// widen the three boxes in quadrature (ring_swept_filter's form, exact on row 0) and longer ones
// difference the kernel rows. Wide penumbrae fold the sweep into the smooth kernel's box.
float ring_penumbra_swept(sampler2DArray table, float slot, float u, float h, float s) {
    float blend = ring_penumbra_smooth_blend(h), value = 0.0;
    if (blend < 1.0) {
        float sharp;
        if (s <= 0.5*h) {
            float h2 = h*h, s2 = s*s;
            sharp = 0.08*ring_penumbra_box(table, slot, u, sqrt(0.1296*h2+s2))
                   +0.42*ring_penumbra_box(table, slot, u, sqrt(0.5184*h2+s2))
                   +0.50*ring_penumbra_box(table, slot, u, sqrt(h2+s2));
        } else sharp = ring_penumbra_rows(table, slot, u, h, s);
        value = (1.0-blend)*sharp;
    }
    if (blend > 0.0) value += blend*ring_penumbra_smooth(table, slot, u, h, s);
    return clamp(value, 0.0, 1.0);
}

float ring_penumbra_extinction(sampler2DArray table, float slot, float u, float h) {
    float blend = ring_penumbra_smooth_blend(h), value = 0.0;
    if (blend < 1.0)
        value = (1.0-blend)*(0.08*ring_penumbra_box(table, slot, u, 0.36*h)+0.42*ring_penumbra_box(table, slot, u, 0.72*h)
                             +0.50*ring_penumbra_box(table, slot, u, h));
    if (blend > 0.0) value += blend*ring_penumbra_smooth(table, slot, u, h, 0.0);
    return clamp(value, 0.0, 1.0);
}

// Decodes a caster's support.w: 0 = no table, otherwise slot + 1 + the slot's mu coordinate.
bool ring_penumbra_slot(float encoded, float mu_coord, out float slot) {
    slot = floor(encoded)-1.0;
    if (slot < 0.0) return false;
    float built = encoded-floor(encoded);
    return abs(mu_coord-built) <= 1e-5+2e-3*built;
}
#endif
