import re
import sys

def main():
    with open('d:/Files/Coding/OpenGL/Stellar-Forge/physics_core.py', 'r') as f:
        content = f.read()

    # 1. Remove import rebound
    content = content.replace("import rebound\n", "")
    
    # 2. Replace _get_particle_array and _extract_render_state
    old_get_particle = """_PARTICLE_STRIDE = None

def _get_particle_array(sim, num_bodies):
    \"\"\"Get a NumPy view of REBOUND's internal C particle array (zero-copy).

    Returns array of shape (num_bodies, 16) where columns are:
    [0]=x [1]=y [2]=z [3]=vx [4]=vy [5]=vz [6..8]=ax,ay,az [9]=m ...
    \"\"\"
    global _PARTICLE_STRIDE
    if _PARTICLE_STRIDE is None:
        _PARTICLE_STRIDE = ctypes.sizeof(rebound.Particle) // 8
    n = _PARTICLE_STRIDE
    addr = ctypes.addressof(sim._particles.contents)
    raw = (ctypes.c_double * (num_bodies * n)).from_address(addr)
    return np.frombuffer(raw, dtype=np.float64).reshape(num_bodies, n)

def _extract_render_state(sim, num_bodies, out_pos, out_vel):
    \"\"\"Extract particle positions/velocities with render coordinate swap (x, z, -y).\"\"\"
    arr = _get_particle_array(sim, num_bodies)
    out_pos[:, 0] = arr[:, 0]
    out_pos[:, 1] = arr[:, 2]
    out_pos[:, 2] = -arr[:, 1]
    out_vel[:, 0] = arr[:, 3]
    out_vel[:, 1] = arr[:, 5]
    out_vel[:, 2] = -arr[:, 4]"""
    
    new_extract_state = """def _extract_render_state(sim, num_bodies, out_pos, out_vel):
    \"\"\"Extract particle positions/velocities with render coordinate swap (x, z, -y).\"\"\"
    out_pos[:, 0] = sim.arr[:num_bodies, 0]
    out_pos[:, 1] = sim.arr[:num_bodies, 2]
    out_pos[:, 2] = -sim.arr[:num_bodies, 1]
    out_vel[:, 0] = sim.arr[:num_bodies, 3]
    out_vel[:, 1] = sim.arr[:num_bodies, 5]
    out_vel[:, 2] = -sim.arr[:num_bodies, 4]"""
    
    content = content.replace(old_get_particle, new_extract_state)
    
    # 3. Add compute_base_gravity
    old_attach = """def attach_custom_forces(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list):
    if not has_j2 and not has_gr:
        return
        
    stride = ctypes.sizeof(rebound.Particle) // 8
    G_val = sim.G
    c2 = C_AU_YR * C_AU_YR
    
    oblate_indices = None
    oblate_j2 = None
    oblate_j4 = None
    oblate_req = None
    oblate_poles = None
    
    if has_j2 and oblate_physics_list:
        oblate_indices = np.array([x[0] for x in oblate_physics_list], dtype=np.int32)
        oblate_j2 = np.array([x[1] for x in oblate_physics_list], dtype=np.float64)
        oblate_j4 = np.array([x[2] for x in oblate_physics_list], dtype=np.float64)
        oblate_req = np.array([x[3] for x in oblate_physics_list], dtype=np.float64)
        oblate_poles = np.array([x[4] for x in oblate_physics_list], dtype=np.float64)
        
    initial_addr = ctypes.addressof(sim._particles.contents)
    
    def force_callback(sim_ref):
        n = sim.N
        raw = (ctypes.c_double * (n * stride)).from_address(initial_addr)
        arr = np.frombuffer(raw, dtype=np.float64).reshape(n, stride)
        compute_custom_forces(arr, n, G_val, c2, has_j2, has_gr, phys_star_idx, 
                              oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles)

    sim.additional_forces = force_callback
    sim.force_is_velocity_dependent = 1"""

    base_gravity = """@njit(cache=True)
def compute_all_accelerations(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, 
                              oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles):
    # Zero out accelerations
    for i in range(num_bodies):
        arr[i, 6] = 0.0
        arr[i, 7] = 0.0
        arr[i, 8] = 0.0
        
    # Base N-body Newtonian Gravity
    for i in range(num_bodies):
        xi = arr[i, 0]
        yi = arr[i, 1]
        zi = arr[i, 2]
        for j in range(i + 1, num_bodies):
            dx = arr[j, 0] - xi
            dy = arr[j, 1] - yi
            dz = arr[j, 2] - zi
            r2 = dx*dx + dy*dy + dz*dz
            if r2 == 0.0: continue
            r_inv = 1.0 / math.sqrt(r2)
            r3_inv = r_inv * r_inv * r_inv
            
            f = G_val * r3_inv
            
            mj = arr[j, 9]
            if mj > 0.0:
                arr[i, 6] += f * mj * dx
                arr[i, 7] += f * mj * dy
                arr[i, 8] += f * mj * dz
                
            mi = arr[i, 9]
            if mi > 0.0:
                arr[j, 6] -= f * mi * dx
                arr[j, 7] -= f * mi * dy
                arr[j, 8] -= f * mi * dz

    compute_custom_forces(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, 
                          oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles)

def attach_custom_forces(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list):
    sim.has_j2 = has_j2
    sim.has_gr = has_gr
    sim.phys_star_idx = phys_star_idx
    if has_j2 and oblate_physics_list:
        sim.oblate_indices = np.array([x[0] for x in oblate_physics_list], dtype=np.int32)
        sim.oblate_j2 = np.array([x[1] for x in oblate_physics_list], dtype=np.float64)
        sim.oblate_j4 = np.array([x[2] for x in oblate_physics_list], dtype=np.float64)
        sim.oblate_req = np.array([x[3] for x in oblate_physics_list], dtype=np.float64)
        sim.oblate_poles = np.array([x[4] for x in oblate_physics_list], dtype=np.float64)
    else:
        sim.oblate_indices = np.empty(0, dtype=np.int32)
        sim.oblate_j2 = np.empty(0, dtype=np.float64)
        sim.oblate_j4 = np.empty(0, dtype=np.float64)
        sim.oblate_req = np.empty(0, dtype=np.float64)
        sim.oblate_poles = np.empty((0, 3), dtype=np.float64)
"""
    content = content.replace(old_attach, base_gravity)
    
    with open('d:/Files/Coding/OpenGL/Stellar-Forge/physics_core.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    main()
