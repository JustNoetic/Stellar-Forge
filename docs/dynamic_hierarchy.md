# Dynamic reference hierarchy

`Simulation` publishes an immutable `ReferenceHierarchy` through `SystemSnapshot`.
It contains parent indices, distance-sorted child lists and root indices. Body
indices never change when an object moves between parents or changes sibling
order. Outliner selection, tracking and expansion use those stable indices.

The reference hierarchy refreshes at 15 Hz of successful, unpaused frame time,
independently of simulation warp, direction and rendering frame rate. A slow
frame performs one refresh using its final state; there is no catch-up loop.
System loading, accepted physical edits and motion-model changes refresh
immediately. Pausing flushes any pending state. Failed steps and idle paused
frames do not refresh. The interval is defined in `Simulation::advance`.

## Parent selection

IAS15 mode classifies reference parents from current positions. Analytical and
SPICE modes retain their declared parents and still sort displayed siblings by
live distance. This feature does not implement analytical patched-conic
transitions or rewrite saved orbital seeds.

Declared roots remain anchors. Other physical bodies can parent less massive
bodies, with body index breaking equal-mass ties. Spacecraft can enter any
positive-mass physical body's region and cannot become parents. This ordering
prevents cycles without changing body storage order. A body outside all finite
regions falls back to its declared root; if that root is no longer an eligible
parent after a mass edit, the most massive eligible body is the fallback.

Candidate regions use the instantaneous Hill estimate
`distance_to_reference_parent * cbrt(body_mass / parent_mass / 3)`, with physical
radius as a minimum. Entering requires distance below 90% of that radius; the
current parent remains eligible out to 110%. Among eligible containing regions,
the smallest radius wins, then normalized distance and body index break ties.
This favors local moon references inside planetary regions. It is a reference
frame policy, not a test of gravitational capture, long-term stability or binary
membership. Equal-mass binaries keep a deterministic physical-body reference;
automatic barycenter nodes are not created.

`ReferenceHierarchyTracker` builds a balanced AABB tree over possible finite
regions and processes physical bodies from larger to smaller mass. Each selected
parent determines that body's region, which is refitted into the tree before
its possible children are queried. Thus a whole subsystem can change host in
one update, and other smaller objects can enter its updated regions immediately.
Queries consider containing volumes rather than all body pairs. Tree and query
scratch buffers are reused. Typical work is O(N log N + candidate visits), plus
sibling sorting; heavily overlapping volumes can still require many visits.

## Physics and consumers

`BodyDefinition::parent_idx` and `jacobi_hierarchy()` remain the analytical
propagation hierarchy. `reference_parent()` identifies the current display/orbit
reference. `orbit_hierarchy()` provides Jacobi coordinates for that reference
topology, preserving numerical sibling order independently of display sorting.
Its explicit traversal supports a reference parent with a later body index.

IAS15 forces, physical positions/velocities, stored orbital seeds and the
definition-based tidal-lock model are unchanged by reference classification.
Outliner nodes, orbit curves, inspector orbit measurements and barycenter camera
views use the published reference topology. Orbit edits use the displayed
parent; a draft requesting an orbit change is rejected if that parent changed
since the draft was prepared. Moving a body's descendants follows reference
child lists rather than assuming descendants have higher array indices.

Reference data is published only when topology or display order changes.
`hierarchy_revision()` invalidates hierarchy-dependent rendering, including a
pause flush that changes no physical state revision. Reference Jacobi masses are
rebuilt on dynamics edits. Captured snapshots retain their previous hierarchy.

`StellarHierarchyTests` checks entry/exit margins, escape and encounters, moved
subsystems, equal-mass cycle prevention, later-index parents, current orbit
frames, stale orbit drafts, snapshot isolation, cadence, reverse time, pause
flushing and an 8,193-body spatial-culling fixture. The fixture prints CPU update
time and narrow-phase candidate count; it is not an N-body integration benchmark.
