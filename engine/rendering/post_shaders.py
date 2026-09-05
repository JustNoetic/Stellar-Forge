from engine.rendering.shader_loader import load_shader

bloom_downsample_shader_vs = load_shader("post/fullscreen_quad.vert")
bloom_downsample_shader_fs = load_shader("post/bloom_downsample.frag")

bloom_upsample_shader_vs = bloom_downsample_shader_vs
bloom_upsample_shader_fs = load_shader("post/bloom_upsample.frag")

composite_shader_vs = bloom_downsample_shader_vs
composite_shader_fs = load_shader("post/composite.frag")

atmo_composite_shader_vs = bloom_downsample_shader_vs
atmo_composite_shader_fs = load_shader("post/atmo_composite.frag")

accum_shader_vs = bloom_downsample_shader_vs
accum_shader_fs = load_shader("post/accum.frag")

star_streak_shader_vs = bloom_downsample_shader_vs
star_streak_shader_fs = load_shader("post/star_streak.frag")

# FFT Convolution Bloom compute shaders
conv_bloom_common_src = load_shader("post/conv_bloom_common.glsl")
conv_bloom_kernel_src = load_shader("post/conv_bloom_kernel.comp")
conv_bloom_scene_src = load_shader("post/conv_bloom_scene.comp")
conv_bloom_convolve_src = load_shader("post/conv_bloom_convolve.comp")

conv_bloom_krow_shader = f"#version 430\n#define ROW\n{conv_bloom_common_src}\n{conv_bloom_kernel_src}"
conv_bloom_kcol_shader = f"#version 430\n#define COL\n{conv_bloom_common_src}\n{conv_bloom_kernel_src}"

conv_bloom_srow_shader = f"#version 430\n#define ROW\n{conv_bloom_common_src}\n{conv_bloom_scene_src}"
conv_bloom_scol_shader = f"#version 430\n#define COL\n{conv_bloom_common_src}\n{conv_bloom_scene_src}"

conv_bloom_crow_shader = f"#version 430\n#define ROW\n#define IFFT\n{conv_bloom_common_src}\n{conv_bloom_convolve_src}"
conv_bloom_ccol_shader = f"#version 430\n#define COL\n#define IFFT\n{conv_bloom_common_src}\n{conv_bloom_convolve_src}"

