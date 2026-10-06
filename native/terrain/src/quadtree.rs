use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cell {
    pub face: u32,
    pub lod: u32,
    pub x: u32,
    pub y: u32,
}
impl Cell {
    pub fn parent(self) -> Self {
        Self {
            lod: self.lod - 1,
            x: self.x / 2,
            y: self.y / 2,
            ..self
        }
    }
    pub fn children(self) -> [Self; 4] {
        std::array::from_fn(|i| Self {
            lod: self.lod + 1,
            x: self.x * 2 + (i as u32 & 1),
            y: self.y * 2 + (i as u32 >> 1),
            ..self
        })
    }
    pub fn contains(self, other: Self) -> bool {
        self.face == other.face
            && self.lod <= other.lod
            && (other.x >> (other.lod - self.lod)) == self.x
            && (other.y >> (other.lod - self.lod)) == self.y
    }
    pub fn range(self) -> [f64; 4] {
        let s = 2. / (1u32 << self.lod) as f64;
        let u = -1. + self.x as f64 * s;
        let v = -1. + self.y as f64 * s;
        [u, v, u + s, v + s]
    }
    pub fn packed(self) -> u32 {
        self.face | (self.lod << 3) | (self.x << 7) | (self.y << 18)
    }
}
pub fn direction(face: u32, u: f64, v: f64) -> [f64; 3] {
    let x = (u * std::f64::consts::FRAC_PI_4).tan();
    let y = (v * std::f64::consts::FRAC_PI_4).tan();
    let p = match face {
        0 => [1., -y, -x],
        1 => [-1., -y, x],
        2 => [x, 1., y],
        3 => [x, -1., -y],
        4 => [x, -y, 1.],
        _ => [-x, -y, -1.],
    };
    let inv = 1. / dot(p, p).sqrt();
    p.map(|a| a * inv)
}
pub fn face_uv(p: [f64; 3]) -> (u32, [f64; 2]) {
    let [x, y, z] = p;
    let (face, a, b) = if x.abs() >= y.abs() && x.abs() >= z.abs() {
        (
            if x >= 0. { 0 } else { 1 },
            if x >= 0. { -z / x.abs() } else { z / x.abs() },
            -y / x.abs(),
        )
    } else if y.abs() >= z.abs() {
        (
            if y >= 0. { 2 } else { 3 },
            x / y.abs(),
            if y >= 0. { z / y.abs() } else { -z / y.abs() },
        )
    } else {
        (
            if z >= 0. { 4 } else { 5 },
            if z >= 0. { x / z.abs() } else { -x / z.abs() },
            -y / z.abs(),
        )
    };
    (
        face,
        [
            a.atan() / std::f64::consts::FRAC_PI_4 * 0.5 + 0.5,
            b.atan() / std::f64::consts::FRAC_PI_4 * 0.5 + 0.5,
        ],
    )
}
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub struct View {
    pub camera: [f64; 3],
    pub radius: f64,
    pub obl: f64,
    pub pixels: f64,
    pub tan_fov: f64,
    pub threshold: f64,
    pub max_lod: u32,
    pub budget: usize,
    pub planes: Vec<[f64; 4]>,
    pub cloud: f64,
    pub bend: f64,
    pub height: f64,
    pub altitude: Option<f64>,
}
#[derive(Clone)]
struct Patch {
    cell: Cell,
    size: f64,
    score: f64,
    full_children: bool,
}
impl PartialEq for Patch {
    fn eq(&self, b: &Self) -> bool {
        self.cell == b.cell
    }
}
impl Eq for Patch {}
impl PartialOrd for Patch {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for Patch {
    fn cmp(&self, b: &Self) -> Ordering {
        self.score.total_cmp(&b.score).then(self.cell.cmp(&b.cell))
    }
}
impl View {
    fn evaluate(&self, cell: Cell) -> Option<Patch> {
        let [u, v, upper_u, upper_v] = cell.range();
        let n = direction(cell.face, (u + upper_u) * 0.5, (v + upper_v) * 0.5);
        let shape = |d: [f64; 3]| {
            [
                d[0] * self.radius,
                d[1] * (1. - self.obl) * self.radius,
                d[2] * self.radius,
            ]
        };
        let center = shape(n);
        let mut size: f64 = 0.;
        for (a, b) in [(u, v), (upper_u, v), (u, upper_v), (upper_u, upper_v)] {
            let p = shape(direction(cell.face, a, b));
            size = size.max(
                dot(
                    std::array::from_fn(|i| p[i] - center[i]),
                    std::array::from_fn(|i| p[i] - center[i]),
                )
                .sqrt(),
            );
        }
        size *= 1.05;
        let extra = self.cloud.max(0.) + self.height.max(0.);
        let bound = size + extra;
        let c = [
            self.camera[0],
            self.camera[1] / (1. - self.obl),
            self.camera[2],
        ];
        let distance = dot(c, c).sqrt().max(1e-9);
        let mut horizon = (self.radius / (distance.max(self.radius + 1e-4)))
            .min(1.)
            .acos();
        if self.cloud > 0. {
            horizon += (self.radius / (self.radius + self.cloud)).min(1.).acos();
        }
        horizon += (self.bend * 0.5).clamp(0., 0.05);
        let angle = horizon + (bound / self.radius).min(1.).asin();
        if angle < std::f64::consts::PI && dot(n, c) / distance < angle.cos() {
            return None;
        }
        // A spherical-cap distance stays meaningful when the camera is inside a
        // patch's bounding sphere. Elevation bounds are for visibility only;
        // subtracting the planet-wide mountain height collapsed kilometre-wide
        // regions to the same near-zero distance and destabilized priorities.
        let separation = (dot(n, c) / distance).clamp(-1., 1.).acos();
        let cap = (size / (self.radius * (1. - self.obl)) / 2.).min(1.).asin() * 2.;
        let gap = (separation - cap).max(0.);
        let altitude = self.altitude.unwrap_or(distance - self.radius).abs();
        let closest =
            (altitude * altitude + 4. * self.radius * distance * (gap * 0.5).sin().powi(2)).sqrt();
        let score = size * self.pixels / (closest.max(0.01) * 2. * self.tan_fov);
        Some(Patch {
            cell,
            size,
            score,
            full_children: separation + cap <= horizon,
        })
    }

    pub fn visible(&self, cell: Cell, size: f64) -> bool {
        let extra = self.height.max(0.) + self.cloud.max(0.);
        self.visible_with_heights(cell, size, -extra, extra)
    }

    pub fn visible_with_heights(&self, cell: Cell, size: f64, minimum: f64, maximum: f64) -> bool {
        if self.planes.is_empty() {
            return true;
        }
        let [u, v, upper_u, upper_v] = cell.range();
        let n = direction(cell.face, (u + upper_u) * 0.5, (v + upper_v) * 0.5);
        let center = [
            n[0] * self.radius,
            n[1] * (1. - self.obl) * self.radius,
            n[2] * self.radius,
        ];
        let bound = size + self.height.max(0.) + self.cloud.max(0.);
        let delta = std::array::from_fn(|i| center[i] - self.camera[i]);
        let padding = self.radius * 2e-7
            + 2. * (self.bend.clamp(0., 0.05) * 0.5).sin() * (dot(delta, delta).sqrt() + bound);
        // Cheap sphere rejection first. A sphere containing the eye is not a
        // reason to submit the patch: the curved surface can still be behind it.
        if self.planes.iter().any(|p| {
            dot(center, [p[0], p[1], p[2]]) + p[3]
                < -(bound + padding) * dot([p[0], p[1], p[2]], [p[0], p[1], p[2]]).sqrt()
        }) {
            return false;
        }
        // Bound the actual curved patch, including radial displacement, rather
        // than expanding its tangential width in every direction. Cube-face
        // rectangles are cones spanned by their four corners; these angular
        // caps therefore contain every point, including stitched edge chords.
        let radial = center.map(|a| a / dot(center, center).sqrt());
        let mut cos_cap: f64 = 1.;
        let mut cos_radial: f64 = 1.;
        for (a, b) in [(u, v), (upper_u, v), (u, upper_v), (upper_u, upper_v)] {
            let d = direction(cell.face, a, b);
            cos_cap = cos_cap.min(dot(n, d));
            let shaped = [d[0], d[1] * (1. - self.obl), d[2]];
            cos_radial = cos_radial.min(dot(radial, shaped) / dot(shaped, shaped).sqrt());
        }
        !self.planes.iter().any(|p| {
            let normal = [p[0], p[1], p[2]];
            let shaped_normal = [p[0], p[1] * (1. - self.obl), p[2]];
            let surface = self.radius * cap_support(shaped_normal, n, cos_cap);
            let upper = cap_support(normal, radial, cos_radial);
            let lower = -cap_support(normal.map(|a| -a), radial, cos_radial);
            let displacement = (minimum * lower)
                .max(minimum * upper)
                .max(maximum * lower)
                .max(maximum * upper);
            // Match float32 shader radius/rotation precision without kilometre
            // overscan. Elevation itself is bounded in its radial direction.
            let padding = padding * dot(normal, normal).sqrt();
            surface + displacement + p[3] < -padding
        })
    }
}

fn cap_support(normal: [f64; 3], axis: [f64; 3], cos_cap: f64) -> f64 {
    let length = dot(normal, normal).sqrt();
    let along = dot(normal, axis);
    if along >= length * cos_cap {
        length
    } else {
        along * cos_cap
            + (length * length - along * along).max(0.).sqrt()
                * (1. - cos_cap * cos_cap).max(0.).sqrt()
    }
}
#[cfg(test)]
fn edge_point(c: Cell, edge: usize, t: f64) -> [f64; 3] {
    let [u, v, upper_u, upper_v] = c.range();
    let eps = (upper_u - u) * 1e-5;
    let (a, b) = match edge {
        0 => (u - eps, v + (upper_v - v) * t),
        1 => (upper_u + eps, v + (upper_v - v) * t),
        2 => (u + (upper_u - u) * t, v - eps),
        _ => (u + (upper_u - u) * t, upper_v + eps),
    };
    direction(c.face, a, b)
}
#[cfg(test)]
fn covering(leaves: &HashMap<Cell, Patch>, p: [f64; 3], max_lod: u32) -> Option<Cell> {
    let (face, uv) = face_uv(p);
    for lod in 0..=max_lod {
        let n = 1u32 << lod;
        let c = Cell {
            face,
            lod,
            x: (uv[0].clamp(0., 1. - 1e-12) * n as f64) as u32,
            y: (uv[1].clamp(0., 1. - 1e-12) * n as f64) as u32,
        };
        if leaves.contains_key(&c) {
            return Some(c);
        }
    }
    None
}

fn coarse_neighbor(leaves: &HashMap<Cell, Patch>, c: Cell, edge: usize) -> Option<Cell> {
    let count = 1 << c.lod;
    let mut neighbor = match edge {
        0 if c.x > 0 => Cell { x: c.x - 1, ..c },
        1 if c.x + 1 < count => Cell { x: c.x + 1, ..c },
        2 if c.y > 0 => Cell { y: c.y - 1, ..c },
        3 if c.y + 1 < count => Cell { y: c.y + 1, ..c },
        _ => {
            // Only cube-face boundary edges require a spherical mapping. On
            // each face, integer adjacency is exact at every supported depth.
            let [u, v, upper_u, upper_v] = c.range();
            let eps = (upper_u - u) * 1e-5;
            let (a, b) = match edge {
                0 => (u - eps, (v + upper_v) * 0.5),
                1 => (upper_u + eps, (v + upper_v) * 0.5),
                2 => ((u + upper_u) * 0.5, v - eps),
                _ => ((u + upper_u) * 0.5, upper_v + eps),
            };
            let (face, uv) = face_uv(direction(c.face, a, b));
            Cell {
                face,
                lod: c.lod,
                x: (uv[0].clamp(0., 1. - 1e-12) * count as f64) as u32,
                y: (uv[1].clamp(0., 1. - 1e-12) * count as f64) as u32,
            }
        }
    };
    // Balancing/edge masks only need equal or coarser neighbours. Start at
    // this depth, where a balanced mesh normally resolves in one or two probes.
    loop {
        if leaves.contains_key(&neighbor) {
            return Some(neighbor);
        }
        if neighbor.lod == 0 {
            return None;
        }
        neighbor = neighbor.parent();
    }
}
pub fn traverse(view: &View) -> Vec<f32> {
    traverse_with_history(view, &[])
}

pub fn traverse_with_history(view: &View, previous: &[Cell]) -> Vec<f32> {
    let mut branches = HashSet::new();
    for cell in previous {
        let mut c = *cell;
        while c.lod > 0 {
            c = c.parent();
            branches.insert(c);
        }
    }
    let mut leaves = HashMap::new();
    let mut candidates = BinaryHeap::new();
    for face in 0..6 {
        if let Some(p) = view.evaluate(Cell {
            face,
            lod: 0,
            x: 0,
            y: 0,
        }) {
            candidates.push(p);
        }
    }
    // Preserve whole parents when a split would exceed the budget. No truncated
    // depth-first traversal that silently drops the remaining visible surface.
    while leaves.len() < view.budget {
        if let Some(p) = candidates.pop() {
            leaves.insert(p.cell, p);
        } else {
            break;
        }
    }
    candidates = leaves.values().cloned().collect();
    while let Some(p) = candidates.pop() {
        // Existing branches merge at 80% of the split threshold. The gap keeps
        // small motion near a threshold from alternating split/merge frames.
        let threshold = view.threshold * if branches.contains(&p.cell) { 0.8 } else { 1. };
        if p.score <= threshold || p.cell.lod >= view.max_lod || !leaves.contains_key(&p.cell) {
            continue;
        }
        // A patch fully inside the horizon always has four visible children.
        // Avoid calculating their geometry when that split cannot fit anyway.
        if p.full_children && leaves.len() + 3 > view.budget {
            continue;
        }
        let children: Vec<_> = p
            .cell
            .children()
            .into_iter()
            .filter_map(|c| view.evaluate(c))
            .collect();
        if children.is_empty() {
            // A parent's conservative horizon bound can intersect the visible
            // cap even when none of its children do. Keeping that empty parent
            // lets balancing propagate its coarse LOD into visible neighbours.
            leaves.remove(&p.cell);
            continue;
        }
        if leaves.len() - 1 + children.len() > view.budget {
            continue;
        }
        leaves.remove(&p.cell);
        for child in children {
            leaves.insert(child.cell, child.clone());
            candidates.push(child);
        }
    }
    // Balance by coarsening, so balancing cannot exceed the same hard budget.
    for _ in 0..view.max_lod {
        let mut changes = Vec::new();
        for c in leaves.keys() {
            for edge in 0..4 {
                if let Some(n) = coarse_neighbor(&leaves, *c, edge) {
                    if c.lod > n.lod + 1 {
                        let mut a = *c;
                        while a.lod > n.lod + 1 {
                            a = a.parent();
                        }
                        changes.push(a);
                    }
                }
            }
        }
        if changes.is_empty() {
            break;
        }
        changes.sort();
        changes.dedup();
        for a in changes {
            if leaves.keys().any(|c| c.contains(a)) {
                continue;
            }
            if let Some(p) = view.evaluate(a) {
                leaves.retain(|c, _| !a.contains(*c));
                leaves.entry(a).or_insert(p);
            }
        }
    }
    let mut cells: Vec<_> = leaves.keys().copied().collect();
    cells.sort();
    let mut out = Vec::with_capacity(cells.len() * 10);
    for c in cells {
        let p = &leaves[&c];
        // Selection and balancing include off-screen neighbours. Camera
        // rotation only changes visibility, never the refinement competition.
        if !view.visible(c, p.size) {
            continue;
        }
        let mut edges = 0u32;
        for e in 0..4 {
            if coarse_neighbor(&leaves, c, e).is_some_and(|n| n.lod < c.lod) {
                edges |= 1 << e;
            }
        }
        let r = c.range();
        out.extend(r.map(|a| a as f32));
        out.extend([
            c.face as f32,
            c.lod as f32,
            c.x as f32,
            c.y as f32,
            p.size as f32,
            edges as f32,
        ]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn view(budget: usize) -> View {
        View {
            camera: [0., 0., 6371.01],
            radius: 6371.,
            obl: 0.,
            pixels: 1080.,
            tan_fov: 0.4142,
            threshold: 50.,
            max_lod: 10,
            budget,
            planes: vec![],
            cloud: 0.,
            bend: 0.,
            height: 8.,
            altitude: None,
        }
    }
    #[test]
    fn mapping_round_trip() {
        for f in 0..6 {
            for (u, v) in [(-0.8, 0.2), (0., 0.), (0.5, -0.9)] {
                let (face, uv) = face_uv(direction(f, u, v));
                assert_eq!(face, f);
                assert!((uv[0] * 2. - 1. - u).abs() < 1e-12);
                assert!((uv[1] * 2. - 1. - v).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn empty_sky_rejects_curved_surface_even_near_ground() {
        let mut v = view(512);
        for (altitude, height) in [(100., 8.), (10., 8.), (0.01, 0.)] {
            v.camera = [0., 0., v.radius + altitude];
            v.altitude = Some(altitude);
            v.height = height;
            v.planes = vec![[0., 0., 1., -v.camera[2] - 0.001]];
            assert!(
                traverse(&v).is_empty(),
                "altitude={altitude} height={height}"
            );
        }
    }

    #[test]
    fn curved_bounds_keep_displaced_points_on_screen() {
        let mut v = view(512);
        v.height = 9.;
        for obl in [0., 0.1, 0.8] {
            v.obl = obl;
            for face in 0..6 {
                for lod in [0, 3, 12] {
                    let cell = Cell {
                        face,
                        lod,
                        x: (1 << lod) / 3,
                        y: (1 << lod) / 2,
                    };
                    let [u, w, upper_u, upper_w] = cell.range();
                    let size = v.evaluate(cell).map_or(v.radius * 2., |p| p.size);
                    for su in [0., 0.23, 0.5, 0.87, 1.] {
                        for sv in [0., 0.31, 0.5, 0.72, 1.] {
                            let d = direction(face, u + (upper_u - u) * su, w + (upper_w - w) * sv);
                            let shaped = [d[0], d[1] * (1. - obl), d[2]];
                            let len = dot(shaped, shaped).sqrt();
                            for elevation in [-9., 0., 9.] {
                                let point = shaped.map(|a| a * (v.radius + elevation / len));
                                for normal in [
                                    [1., 0., 0.],
                                    [-1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., -1., 0.],
                                    [0., 0., 1.],
                                    [0., 0., -1.],
                                    [0.6, -0.8, 0.],
                                ] {
                                    v.planes = vec![[
                                        normal[0],
                                        normal[1],
                                        normal[2],
                                        -dot(normal, point) + 0.001,
                                    ]];
                                    assert!(
                                        v.visible(cell, size),
                                        "lost {cell:?} obl={obl} elevation={elevation}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn budget_preserves_nonoverlapping_cover() {
        let v = view(64);
        let out = traverse(&v);
        assert!(out.len() / 10 <= 64);
        let cells: Vec<_> = out
            .chunks_exact(10)
            .map(|p| Cell {
                face: p[4] as u32,
                lod: p[5] as u32,
                x: p[6] as u32,
                y: p[7] as u32,
            })
            .collect();
        for a in &cells {
            for b in &cells {
                if a != b {
                    assert!(!a.contains(*b));
                }
            }
        }
    }
    #[test]
    fn neighbors_balanced() {
        let v = view(512);
        let out = traverse(&v);
        let leaves: HashMap<_, _> = out
            .chunks_exact(10)
            .map(|p| {
                let c = Cell {
                    face: p[4] as u32,
                    lod: p[5] as u32,
                    x: p[6] as u32,
                    y: p[7] as u32,
                };
                (
                    c,
                    Patch {
                        cell: c,
                        size: 0.,
                        score: 0.,
                        full_children: false,
                    },
                )
            })
            .collect();
        for c in leaves.keys() {
            for e in 0..4 {
                for t in [0.25, 0.75] {
                    if let Some(n) = covering(&leaves, edge_point(*c, e, t), 10) {
                        assert!(c.lod.abs_diff(n.lod) <= 1, "{c:?} {n:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn integer_neighbors_match_spherical_lookup_across_faces() {
        let mut v = view(512);
        v.max_lod = 16;
        for face in 0..6 {
            for (u, w) in [(0.173, -0.217), (1., 0.217), (1., 1.)] {
                for altitude in [0.01, 10., 1000.] {
                    v.camera = direction(face, u, w).map(|n| n * (v.radius + altitude));
                    v.altitude = Some(altitude);
                    let leaves = cells(&traverse(&v));
                    for c in leaves.keys() {
                        for edge in 0..4 {
                            let fast = coarse_neighbor(&leaves, *c, edge);
                            for t in [0.25, 0.5, 0.75] {
                                let reference =
                                    covering(&leaves, edge_point(*c, edge, t), v.max_lod)
                                        .filter(|n| n.lod <= c.lod);
                                assert_eq!(fast, reference, "{c:?} edge={edge} t={t}");
                            }
                        }
                    }
                }
            }
        }
    }

    fn cells(out: &[f32]) -> HashMap<Cell, Patch> {
        out.chunks_exact(10)
            .map(|p| {
                let c = Cell {
                    face: p[4] as u32,
                    lod: p[5] as u32,
                    x: p[6] as u32,
                    y: p[7] as u32,
                };
                (
                    c,
                    Patch {
                        cell: c,
                        size: p[8] as f64,
                        score: 0.,
                        full_children: false,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn rotation_only_changes_visibility() {
        let mut v = view(512);
        let full = traverse(&v);
        let all = cells(&full);
        for plane in [
            [1., 0., 0., 0.],
            [-1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., -1., 0., 0.],
        ] {
            v.planes = vec![plane];
            let visible = traverse(&v);
            assert!(!visible.is_empty());
            assert!(visible.len() <= full.len());
            for c in cells(&visible).keys() {
                assert!(all.contains_key(c));
            }
        }
    }

    #[test]
    fn descending_keeps_local_detail() {
        let mut v = view(4096);
        v.max_lod = 16;
        v.threshold = 100.;
        let dir = direction(4, 0.173, -0.217);
        let mut previous_lod = 0;
        let mut history = vec![];
        for alt in [100., 10., 1., 0.1, 0.01] {
            v.camera = dir.map(|n| n * (v.radius + alt));
            v.altitude = Some(alt);
            let out = traverse_with_history(&v, &history);
            let leaves = cells(&out);
            assert!(leaves.len() <= v.budget);
            let local = covering(&leaves, dir, v.max_lod).unwrap();
            assert!(
                local.lod >= previous_lod,
                "alt={alt} lod={} previous={previous_lod}",
                local.lod
            );
            previous_lod = local.lod;
            history = leaves.keys().copied().collect();
        }
        assert!(previous_lod >= 14);
    }

    #[test]
    fn hysteresis_retains_branch_until_merge_threshold() {
        let mut v = view(512);
        v.max_lod = 1;
        let root = Cell {
            face: 4,
            lod: 0,
            x: 0,
            y: 0,
        };
        let score = v.evaluate(root).unwrap().score;
        let history = root.children();
        v.threshold = score * 1.1;
        assert!(cells(&traverse(&v)).contains_key(&root));
        assert!(!cells(&traverse_with_history(&v, &history)).contains_key(&root));
        v.threshold = score * 1.3;
        assert!(cells(&traverse_with_history(&v, &history)).contains_key(&root));
    }

    #[test]
    fn horizon_false_positive_does_not_survive_as_coarse_neighbor() {
        let mut v = view(4096);
        v.camera = direction(4, 0.173, -0.217).map(|n| n * (v.radius + 1.));
        v.altitude = Some(1.);
        v.max_lod = 16;
        v.threshold = 75.;
        let leaves = cells(&traverse(&v));
        for c in leaves.keys().filter(|c| c.lod < v.max_lod) {
            let p = v.evaluate(*c).unwrap();
            if p.score > v.threshold {
                assert!(
                    c.children()
                        .iter()
                        .any(|child| v.evaluate(*child).is_some()),
                    "empty horizon parent survived: {c:?}"
                );
            }
        }
    }

    #[test]
    fn camera_travel_preserves_detail_and_balanced_cover() {
        let mut v = view(512);
        v.max_lod = 16;
        v.threshold = 75.;
        // Face interior, patch boundary, cube edge and cube corner. Move both
        // toward the surface and sideways across boundaries with history.
        for (u, w) in [(0.173, -0.217), (0., 0.), (1., 0.217), (1., 1.)] {
            let mut history = vec![];
            for altitude in [100., 10., 1., 0.1, 0.01] {
                for delta in [-1e-5, 0., 1e-5] {
                    let d = direction(4, u + delta, w);
                    v.camera = d.map(|n| n * (v.radius + altitude));
                    v.altitude = Some(altitude);
                    let out = traverse_with_history(&v, &history);
                    let leaves = cells(&out);
                    assert!(leaves.len() <= v.budget);
                    let local = covering(&leaves, d, v.max_lod).expect("camera ground missing");
                    assert!(local.lod >= 6, "altitude={altitude} ground={local:?}");
                    for c in leaves.keys() {
                        for e in 0..4 {
                            for t in [0.25, 0.75] {
                                if let Some(n) = covering(&leaves, edge_point(*c, e, t), v.max_lod)
                                {
                                    assert!(c.lod.abs_diff(n.lod) <= 1, "{c:?} {n:?}");
                                }
                            }
                        }
                    }
                    history = leaves.keys().copied().collect();
                    // A stationary camera must settle on the same mesh, even
                    // with a tight budget and the existing split hysteresis.
                    assert_eq!(out, traverse_with_history(&v, &history));
                }
            }
        }
    }
}
