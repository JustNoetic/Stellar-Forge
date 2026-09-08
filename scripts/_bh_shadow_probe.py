"""Replicate BH shadow silhouette rendering (sphere.vert min-clamp + sphere.frag
capture test) as a function of camera distance, to diagnose the reported
'constant screen-size black sphere' bug."""
import math
import numpy as np

AU_KM = 149597870.7
M_SUN = 100.0                 # Gargantua
RS_KM = 2.95325008 * M_SUN    # 295.3 km
B_C = 2.5980762 * RS_KM       # 767.4 km
BC_AU = B_C / AU_KM           # 5.13e-6 AU

H = 1080.0
FOV_DEG = 45.0
FF = 1.0 / math.tan(math.radians(FOV_DEG) / 2.0)
CLAMP_MIN_PX = 3.0
SPIN = 0.998

def shadow_angular_radius_px(dist_au):
    """Angular radius (px) of the black region: mesh disk radius is BC_AU
    (visual radius); frag capture test: b = |P_min|/metric <= b_c, s_min>0."""
    dist_km = dist_au * AU_KM
    metric = math.sqrt(max(1e-4, 1.0 - RS_KM / max(1e-4, dist_km)))
    # captured: |P_min| <= b_c * metric  (with spin=0)
    b_eff = B_C * metric
    ang_rad = math.asin(min(1.0, b_eff / dist_km))
    return ang_rad * H * FF          # diameter-ish px (matches apparent_px convention)

print(f"rs={RS_KM:.1f} km  b_c={B_C:.1f} km  visual radius={BC_AU:.3e} AU")
print(f"{'cam dist':>14} {'unclamped px':>13} {'rendered disk px':>17}")
for dist_au in [100.0, 10.0, 1.0, 0.1, 0.01, 1e-3, 5e-4, 2e-4, 1e-4, 8e-6, 7e-6, 6.4e-6]:
    app_px = (BC_AU / dist_au) * H * FF
    rendered = max(app_px, CLAMP_MIN_PX)  # min-size clamp in sphere.vert
    # min approach: LAND_ALT 200 km above visual surface
    min_r_au = (B_C + 200.0) / AU_KM
    flag = " <-- below min approach alt" if dist_au < min_r_au else ""
    print(f"{dist_au:14.6g} {app_px:13.3f} {rendered:17.3f}{flag}")
