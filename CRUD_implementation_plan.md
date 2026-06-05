# Realtime CRUD For Stellar-Forge (Remaining Milestones)

We've successfully implemented dynamic memory resizing, object deletion, patched conics, and object editing! Now we are returning to the core CRUD feature to complete the Sandbox experience.

## User Review Required
Please review the final Phased Milestones below to ensure they match your vision for adding objects and editing atmospheres/rings.

## Current Progress Status
- `[x]` **Milestone 1**: The Memory Foundation (Dynamic Buffers)
- `[x]` **Milestone 2**: Object Deletion
- `[x]` **Milestone 3**: Patched Conics (Escape Trajectories)
- `[x]` **Milestone 4**: Object Editing (Mass, Radius, Orbit)

---

## Remaining Milestones

### Milestone 5: Object Creation (Add Body)
We will add a dedicated "Add Body" menu.
- **UI Menu:** A new window (or section in the Inspector) to specify the new body's Name, Mass, Radius, Color, and Orbit parameters.
- **Parent Selection:** A dropdown to select which existing body the new object will orbit.
- **Physics Integration:** Sending the `CREATE` payload to `crud_queue`. The physics thread will compute the Cartesian state relative to the chosen parent, call `sim.add(m=..., x=..., ...)`, and append the new body to all shared arrays.
- **Graphics Integration:** The main thread will intercept the `CREATE` operation, expand the visual arrays (`visual_data`, `bodies_data`, `all_instances`), and seamlessly incorporate it into the rendering loop without dropping frames.

### Milestone 6: Atmosphere & Rings Editor
- **Atmosphere CRUD:** Expand the Edit Mode UI to toggle atmospheres and edit the rendering radius, thickness, and scattering coefficients (Beta Rayleigh/Mie). 
- **Rings CRUD:** Expand the Edit Mode UI to toggle rings and edit inner/outer radius, opacity, and color gradients.
- **Dynamic Geometry Regeneration:** The engine currently precomputes ring geometry once at startup (`ring_precomputed`). We will refactor this to allow a specific body's `ring_vbo` and `ring_ibo` to be dynamically discarded and rebuilt when you apply ring edits in the UI.

## Verification Plan
For Object Creation, we will verify by creating a new moon around Pluto and ensuring the arrays do not desync. For the Atmosphere & Rings Editor, we will edit Saturn's rings dynamically to verify the geometry regenerates safely without stalling the main thread.
