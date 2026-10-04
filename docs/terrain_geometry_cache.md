# Cached terrain geometry

The CPU quadtree still selects and packs ground terrain patches. The draw shader
uses cached body-local elevation and normals, instead of re-evaluating procedural
elevation five times per vertex on every frame. Clouds retain their existing path.

`TerrainGeometryCache.prepare` selects reusable entries and submits up to 128
changed patches per frame. A height compute pass evaluates one elevation per grid
point. A normal pass shares those heights between neighbouring vertices, preserving
the existing central differences and one-sided patch-boundary differences. Skirts
reuse their surface vertex and apply the existing extrusion during drawing.
Unbaked or uncached patches use the original vertex path, so cache pressure cannot
remove terrain or temporarily flatten it.

Cache identity includes body name, patch range, face, LOD, grid resolution, height
slot and ancestor UV mapping, elevation range, radius and oblateness. Each texture
slot has an upload generation, so replacement, editing or recycling of a height
layer invalidates its geometry. Camera motion, spin, lighting and diffuse-only
uploads do not invalidate cached geometry. An unchanged, fully cached cut skips
per-patch lookup and buffer uploads. Every currently referenced entry is protected
before LRU eviction starts.

Geometry storage is capped at 96 MiB. Each vertex stores a float4 (normal XYZ,
combined elevation in kilometres). At the default 32-cell grid this allows 5,777
patch slots; denser grids hold fewer entries and can use the original draw path
for the remainder. Generation scratch storage and the draw mapping are separate,
small bounded buffers. Grid resolution changes recreate the pool. Resources are
released at application shutdown. There are no production GPU readbacks, fences
or explicit GPU completion waits.

Run `python scripts/test_terrain_geometry_cache.py` using the project's normal
dependencies and an OpenGL 4.6 driver. Tests compare cached geometry against the
live vertex path across all six faces, LODs 0/4/6/10, two oblate body shapes, and
all supported grid resolutions. They also cover height-layer revisions, ancestor
UV changes, shape changes, bounded generation, eviction, fallback, cloud shells,
ocean clipping and zero height span. The production terrain program is compiled.

Measured on an RTX 4060 Ti with 1,024 synthetic mountainous LOD-8 patches and
2,359,296 submitted triangles: median vertex-only GPU draw time was 13.93 ms for
the original shader and 0.29 ms after caching. The unchanged-cut CPU cache check
took 0.049 ms. The 128-patch generation cap warmed this cut in eight frames.
An illuminated 768-square image comparison differed by at most one 8-bit channel
level. Transform-feedback comparison found identical positions in tested cases;
maximum normal angular differences were 0.22 degrees from shared-grid floating
point evaluation order. These are isolated measurements, not full-application FPS
guarantees. A changing view or height residency can retain more live shader work.
