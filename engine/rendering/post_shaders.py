from engine.rendering.shader_loader import load_shader

bloom_downsample_shader_vs = load_shader("post/fullscreen_quad.vert")
bloom_downsample_shader_fs = load_shader("post/bloom_downsample.frag")

bloom_upsample_shader_vs = bloom_downsample_shader_vs
bloom_upsample_shader_fs = load_shader("post/bloom_upsample.frag")

composite_shader_vs = bloom_downsample_shader_vs
composite_shader_fs = load_shader("post/composite.frag")

taa_resolve_shader_vs = bloom_downsample_shader_vs
taa_resolve_shader_fs = load_shader("post/taa_resolve.frag")

atmo_composite_shader_vs = bloom_downsample_shader_vs
atmo_composite_shader_fs = load_shader("post/atmo_composite.frag")
