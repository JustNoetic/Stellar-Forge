#version 460

layout(location = 0) in vec3 f_world_pos;
layout(location = 1) flat in uint f_body_idx;
layout(location = 2) in vec3 f_body_offset;
layout(location = 3) in vec3 f_refraction_shift;

#include "common/celestial_scene.glsl"
#include "common/body_surface.glsl"
#include "common/cloud_shadow.glsl"
#include "common/cloud_ringshine.glsl"
#include "common/surface_atmosphere.glsl"
#include "common/fluid_surface.glsl"
#ifdef TRANSPARENCY_CAPTURE
#include "common/transparency.glsl"
#endif
#ifdef CLOUD_SHEET_INLINE
layout(rgba32f, set = 0, binding = 35) uniform writeonly image2D cloud_layer_color;
layout(r32f, set = 0, binding = 36) uniform writeonly image2D cloud_layer_depth;
#endif

layout(location = 0) out vec4 out_color;
const uint CLOUD_MAX_STARS = 4u;

#include "common/cloud_sheet.glsl"

void main() {
    render_cloud_sheet();
}
