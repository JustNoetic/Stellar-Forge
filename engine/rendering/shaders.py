from engine.rendering.shader_loader import load_shader

culling_compute_shader = load_shader("compute/culling.comp")

sphere_vertex_shader = load_shader("celestial/sphere.vert")
sphere_fragment_shader = load_shader("celestial/sphere.frag")

point_celestial_vertex_shader = load_shader("celestial/point_celestial.vert")
point_celestial_fragment_shader = load_shader("celestial/point_celestial.frag")

starfield_vertex_shader = load_shader("celestial/starfield.vert")
starfield_fragment_shader = load_shader("celestial/starfield.frag")



orbit_compute_shader = load_shader("compute/orbit.comp")
orbit_vertex_shader = load_shader("celestial/orbit.vert")
orbit_fragment_shader = load_shader("celestial/orbit.frag")

ephem_orbit_vertex_shader = load_shader("celestial/ephem_orbit.vert")
ephem_orbit_fragment_shader = load_shader("celestial/ephem_orbit.frag")

ring_vertex_shader = load_shader("celestial/ring.vert")
ring_fragment_shader = load_shader("celestial/ring.frag")

hz_vertex_shader = load_shader("celestial/hz.vert")
hz_fragment_shader = load_shader("celestial/hz.frag")

atmo_vertex_shader = load_shader("atmosphere/atmo.vert")
atmo_fragment_shader = load_shader("atmosphere/atmo.frag")

atmo_lut_vertex_shader = load_shader("atmosphere/atmo_lut.vert")
atmo_lut_fragment_shader = load_shader("atmosphere/atmo_lut.frag")

multi_scatter_lut_fragment_shader = load_shader("atmosphere/multi_scatter_lut.frag")

ringshine_map_vertex_shader = load_shader("post/ringshine_map.vert")
ringshine_map_fragment_shader = load_shader("post/ringshine_map.frag")

atmo_upsample_fragment_shader = load_shader("atmosphere/atmo_upsample.frag")

