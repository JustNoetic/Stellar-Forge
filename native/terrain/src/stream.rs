use crate::quadtree::{direction, face_uv, Cell};
use image::ImageReader;
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
};

pub const GUTTER: usize = 2;
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key {
    pub body: String,
    pub map: String,
    pub cell: Cell,
}
#[derive(Default)]
struct Catalog {
    files: HashMap<Key, PathBuf>,
    max: HashMap<(String, String, u32), u32>,
}
fn entries(path: &Path) -> Vec<std::fs::DirEntry> {
    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .collect()
}
fn scan(base: &Path) -> Catalog {
    let mut cat = Catalog::default();
    for body in entries(base) {
        if !body.path().is_dir() {
            continue;
        }
        let b = body.file_name().to_string_lossy().to_lowercase();
        for map in entries(&body.path()) {
            let m = map.file_name().to_string_lossy().to_lowercase();
            if !map.path().is_dir() {
                continue;
            }
            for face in entries(&map.path()) {
                let Ok(f) = face.file_name().to_string_lossy().parse::<u32>() else {
                    continue;
                };
                if f > 5 {
                    continue;
                }
                for level in entries(&face.path()) {
                    let Ok(l) = level.file_name().to_string_lossy().parse::<u32>() else {
                        continue;
                    };
                    if l > 10 {
                        continue;
                    }
                    for file in entries(&level.path()) {
                        let path = file.path();
                        let ext = path
                            .extension()
                            .and_then(|p| p.to_str())
                            .unwrap_or("")
                            .to_ascii_lowercase();
                        if !["png", "jpg", "jpeg", "webp"].contains(&ext.as_str())
                            || (m == "height" && ext != "png")
                        {
                            continue;
                        }
                        let Some((a, bx)) = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .and_then(|s| s.split_once('_'))
                        else {
                            continue;
                        };
                        let (Ok(x), Ok(y)) = (a.parse::<u32>(), bx.parse::<u32>()) else {
                            continue;
                        };
                        if x >= 1 << l || y >= 1 << l {
                            continue;
                        }
                        let key = Key {
                            body: b.clone(),
                            map: m.clone(),
                            cell: Cell {
                                face: f,
                                lod: l,
                                x,
                                y,
                            },
                        };
                        // PNG wins over a stale JPEG left by an earlier bake.
                        if ext == "png" || !cat.files.contains_key(&key) {
                            cat.files.insert(key, path);
                        }
                        cat.max
                            .entry((b.clone(), m.clone(), f))
                            .and_modify(|v| *v = (*v).max(l))
                            .or_insert(l);
                    }
                }
            }
        }
    }
    cat
}
#[derive(Clone)]
struct Job {
    key: Key,
    priority: u32,
    sequence: u64,
    generation: u64,
    catalog: Arc<Catalog>,
}
impl PartialEq for Job {
    fn eq(&self, b: &Self) -> bool {
        self.sequence == b.sequence
    }
}
impl Eq for Job {}
impl PartialOrd for Job {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for Job {
    fn cmp(&self, b: &Self) -> Ordering {
        b.priority
            .cmp(&self.priority)
            .then(b.sequence.cmp(&self.sequence))
    }
}
enum Pixels {
    Color(Vec<[u8; 4]>),
    Height(Vec<f32>),
}
struct Source {
    width: usize,
    height: usize,
    pixels: Pixels,
    padded: bool,
}
fn source(path: &Path, size: usize, height: bool, cloud: bool) -> Result<Source, String> {
    let mut reader = ImageReader::open(path)
        .map_err(|e| e.to_string())?
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(2048);
    limits.max_image_height = Some(2048);
    limits.max_alloc = Some(32 * 1024 * 1024);
    reader.limits(limits);
    let img = reader.decode().map_err(|e| e.to_string())?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let padded = w == size + 2 * GUTTER && h == w;
    let pixels = if height {
        Pixels::Height(img.to_luma32f().into_raw())
    } else if cloud && !img.color().has_alpha() {
        Pixels::Color(
            img.to_luma8()
                .into_raw()
                .into_iter()
                .map(|a| [255, 255, 255, a])
                .collect(),
        )
    } else {
        Pixels::Color(img.to_rgba8().pixels().map(|p| p.0).collect())
    };
    Ok(Source {
        width: w,
        height: h,
        pixels,
        padded,
    })
}
fn sample_source(src: &Source, uv: [f64; 2], channel: usize, size: usize) -> f32 {
    let p = if src.padded {
        [
            uv[0] * size as f64 + GUTTER as f64 - 0.5,
            uv[1] * size as f64 + GUTTER as f64 - 0.5,
        ]
    } else {
        [
            uv[0] * (src.width - 1) as f64,
            uv[1] * (src.height - 1) as f64,
        ]
    };
    let x = p[0].clamp(0., (src.width - 1) as f64);
    let y = p[1].clamp(0., (src.height - 1) as f64);
    let ix = x.floor() as usize;
    let iy = y.floor() as usize;
    let fx = (x - ix as f64) as f32;
    let fy = (y - iy as f64) as f32;
    let fetch = |x: usize, y: usize| {
        let at = y * src.width + x;
        match &src.pixels {
            Pixels::Height(h) => h[at],
            Pixels::Color(c) => c[at][channel] as f32,
        }
    };
    let nx = (ix + 1).min(src.width - 1);
    let ny = (iy + 1).min(src.height - 1);
    (fetch(ix, iy) * (1. - fx) + fetch(nx, iy) * fx) * (1. - fy)
        + (fetch(ix, ny) * (1. - fx) + fetch(nx, ny) * fx) * fy
}
struct Decoded {
    key: Key,
    generation: u64,
    bytes: Vec<u8>,
    heights: Option<Vec<f32>>,
}
fn decode(job: &Job, size: usize) -> Result<Decoded, String> {
    let height = job.key.map == "height";
    let cloud = job.key.map == "clouds";
    let c = job.key.cell;
    let path = job.catalog.files.get(&job.key).ok_or("missing tile")?;
    let central = source(path, size, height, cloud)?;
    let width = size + GUTTER * 2;
    let mut cache = HashMap::new();
    cache.insert(c, central);
    let channels = if height { 1 } else { 4 };
    let mut values = vec![0f32; width * width * channels];
    let count = (1u32 << c.lod) as f64;
    for y in 0..width {
        for x in 0..width {
            let local = [
                (x as f64 - GUTTER as f64 + 0.5) / size as f64,
                (y as f64 - GUTTER as f64 + 0.5) / size as f64,
            ];
            let (cell, uv) = if local.iter().all(|v| *v >= 0. && *v <= 1.) {
                (c, local)
            } else {
                let u = (c.x as f64 + local[0]) / count * 2. - 1.;
                let v = (c.y as f64 + local[1]) / count * 2. - 1.;
                let (face, global) = face_uv(direction(c.face, u, v));
                let gx = (global[0] * count).clamp(0., count - 1e-9);
                let gy = (global[1] * count).clamp(0., count - 1e-9);
                let nc = Cell {
                    face,
                    lod: c.lod,
                    x: gx.floor() as u32,
                    y: gy.floor() as u32,
                };
                (nc, [gx - gx.floor(), gy - gy.floor()])
            };
            if !cache.contains_key(&cell) {
                let key = Key {
                    cell,
                    ..job.key.clone()
                };
                if let Some(p) = job.catalog.files.get(&key) {
                    if let Ok(src) = source(p, size, height, cloud) {
                        cache.insert(cell, src);
                    }
                }
            }
            // Complete v2 tiles already carry actual cross-face samples. Keep them,
            // including gutters, instead of reconstructing from rounded legacy data.
            let (src, coord) = if cache[&c].padded {
                (&cache[&c], local)
            } else if let Some(s) = cache.get(&cell) {
                (s, uv)
            } else {
                (&cache[&c], [local[0].clamp(0., 1.), local[1].clamp(0., 1.)])
            };
            for channel in 0..channels {
                values[(y * width + x) * channels + channel] =
                    sample_source(src, coord, channel, size);
            }
        }
    }
    let (bytes, heights) = if height {
        (Vec::new(), Some(values))
    } else {
        (
            values
                .iter()
                .map(|v| v.round().clamp(0., 255.) as u8)
                .collect(),
            None,
        )
    };
    Ok(Decoded {
        key: job.key.clone(),
        generation: job.generation,
        bytes,
        heights,
    })
}
#[derive(Default)]
struct Queue {
    jobs: BinaryHeap<Job>,
    ready: VecDeque<Result<Decoded, (Key, u64, String)>>,
    inflight: HashSet<Key>,
    touched: HashMap<Key, u64>,
    generation: u64,
    sequence: u64,
    closed: bool,
}
struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
}
struct Slot {
    key: Option<Key>,
    last: u64,
    locked: bool,
    version: u64,
    height: Option<Vec<f32>>,
}
pub struct Stream {
    base: PathBuf,
    catalog: Arc<Catalog>,
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    slots: Vec<Slot>,
    resident: HashMap<Key, usize>,
    failed: HashSet<Key>,
    frame: u64,
    pub revision: u64,
    pub color_capacity: usize,
    pub size: usize,
    max_jobs: usize,
    pub errors: VecDeque<String>,
}
impl Stream {
    pub fn new(base: PathBuf, capacity: usize, size: usize, workers: usize) -> Self {
        let color_capacity = capacity * 3 / 4;
        let catalog = Arc::new(scan(&base));
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            wake: Condvar::new(),
        });
        let handles = (0..workers)
            .map(|_| {
                let state = shared.clone();
                thread::spawn(move || loop {
                    let job = {
                        let mut q = state.queue.lock().unwrap();
                        while q.jobs.is_empty() && !q.closed {
                            q = state.wake.wait(q).unwrap();
                        }
                        if q.closed {
                            break;
                        }
                        q.jobs.pop().unwrap()
                    };
                    let result =
                        decode(&job, size).map_err(|e| (job.key.clone(), job.generation, e));
                    let mut q = state.queue.lock().unwrap();
                    if q.closed {
                        break;
                    }
                    if job.generation == q.generation {
                        q.ready.push_back(result);
                    } // stale jobs never repopulate edited tiles
                })
            })
            .collect();
        let mut slots: Vec<_> = (0..capacity)
            .map(|_| Slot {
                key: None,
                last: 0,
                locked: false,
                version: 0,
                height: None,
            })
            .collect();
        slots[0].locked = true;
        Self {
            base,
            catalog,
            shared,
            workers: handles,
            slots,
            resident: HashMap::new(),
            failed: HashSet::new(),
            frame: 0,
            revision: 0,
            color_capacity,
            size,
            max_jobs: (64 * 1024 * 1024 / ((size + 4) * (size + 4) * 4)).clamp(8, 128),
            errors: VecDeque::new(),
        }
    }
    pub fn max_lod(&self, body: &str, map: &str) -> i32 {
        (0..6)
            .filter_map(|f| self.catalog.max.get(&(body.to_lowercase(), map.into(), f)))
            .max()
            .map(|v| *v as i32)
            .unwrap_or(-1)
    }
    fn clamp(&self, mut key: Key) -> Key {
        if let Some(max) = self
            .catalog
            .max
            .get(&(key.body.clone(), key.map.clone(), key.cell.face))
        {
            while key.cell.lod > *max {
                key.cell = key.cell.parent();
            }
        }
        key
    }
    pub fn request(&mut self, key: Key, priority: Option<u32>) {
        let mut key = self.clamp(key);
        let mut lineage = Vec::new();
        loop {
            if self.catalog.files.contains_key(&key) {
                lineage.push(key.clone());
            }
            if key.cell.lod == 0 {
                break;
            }
            key.cell = key.cell.parent();
        }
        for key in lineage.into_iter().rev() {
            if self.resident.contains_key(&key) || self.failed.contains(&key) {
                continue;
            }
            let mut q = self.shared.queue.lock().unwrap();
            q.touched.insert(key.clone(), self.frame);
            if q.closed || q.inflight.len() >= self.max_jobs || q.inflight.contains(&key) {
                continue;
            }
            q.sequence += 1;
            let sequence = q.sequence;
            let generation = q.generation;
            let p = priority.unwrap_or(key.cell.lod);
            q.jobs.push(Job {
                key: key.clone(),
                priority: p,
                sequence,
                generation,
                catalog: self.catalog.clone(),
            });
            q.inflight.insert(key);
            self.shared.wake.notify_one();
        }
    }
    pub fn resolve(&mut self, key: Key) -> [f32; 4] {
        self.request(key.clone(), None);
        let original = key.cell;
        let mut ancestor = self.clamp(key.clone());
        loop {
            if let Some(&slot) = self.resident.get(&ancestor) {
                self.slots[slot].last = self.frame;
                let d = original.lod - ancestor.cell.lod;
                let scale = 1.0 / (1u32 << d) as f32;
                let mask = (1u32 << d) - 1;
                return [
                    slot as f32,
                    scale,
                    (original.x & mask) as f32 * scale,
                    (original.y & mask) as f32 * scale,
                ];
            }
            if ancestor.cell.lod == 0 {
                break;
            }
            ancestor.cell = ancestor.cell.parent();
        }
        [if key.map == "diffuse" { 0. } else { -1. }, 1., 0., 0.]
    }
    pub fn is_resident(&self, key: Key) -> bool {
        self.resident.contains_key(&self.clamp(key))
    }
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        let mut q = self.shared.queue.lock().unwrap();
        // Reprioritize/cancel queued obsolete requests on camera movement. A
        // currently running decode is bounded and may finish, but cannot grow
        // an unbounded trail of jobs behind a fast-moving camera.
        let jobs: Vec<_> = q.jobs.drain().collect();
        for job in jobs {
            if q.touched
                .get(&job.key)
                .copied()
                .unwrap_or(0)
                .saturating_add(3)
                < self.frame
            {
                q.inflight.remove(&job.key);
            } else {
                q.jobs.push(job);
            }
        }
        let active = q.inflight.clone();
        q.touched.retain(|k, _| active.contains(k));
    }
    pub fn poll(&mut self) -> Option<(usize, String, Vec<u8>)> {
        let mut remaining = self.shared.queue.lock().unwrap().ready.len();
        while remaining > 0 {
            remaining -= 1;
            let result = self.shared.queue.lock().unwrap().ready.pop_front()?;
            let decoded = match result {
                Ok(v) => v,
                Err((key, generation, error)) => {
                    let mut q = self.shared.queue.lock().unwrap();
                    q.inflight.remove(&key);
                    if generation == q.generation {
                        self.failed.insert(key);
                        self.errors.push_back(error);
                        while self.errors.len() > 8 {
                            self.errors.pop_front();
                        }
                    }
                    continue;
                }
            };
            let current = self.shared.queue.lock().unwrap().generation;
            if decoded.generation != current {
                continue;
            }
            if self.resident.contains_key(&decoded.key) {
                self.shared
                    .queue
                    .lock()
                    .unwrap()
                    .inflight
                    .remove(&decoded.key);
                continue;
            }
            let range = if decoded.key.map == "height" {
                self.color_capacity..self.slots.len()
            } else {
                1..self.color_capacity
            };
            let slot = range
                .clone()
                .find(|i| self.slots[*i].key.is_none())
                .or_else(|| {
                    range
                        .filter(|i| {
                            !self.slots[*i].locked
                                && self.slots[*i].last.saturating_add(1) < self.frame
                        })
                        .min_by_key(|i| {
                            let s = &self.slots[*i];
                            s.last
                                + (8u64.saturating_sub(s.key.as_ref().unwrap().cell.lod as u64))
                                    * 250
                        })
                });
            let Some(slot) = slot else {
                self.shared
                    .queue
                    .lock()
                    .unwrap()
                    .ready
                    .push_back(Ok(decoded));
                continue;
            };
            self.shared
                .queue
                .lock()
                .unwrap()
                .inflight
                .remove(&decoded.key);
            if let Some(old) = self.slots[slot].key.take() {
                self.resident.remove(&old);
            }
            let locked = self
                .slots
                .iter()
                .filter(|s| {
                    s.locked
                        && s.key
                            .as_ref()
                            .is_some_and(|k| (k.map == "height") == (decoded.key.map == "height"))
                })
                .count();
            let limit = if decoded.key.map == "height" {
                (self.slots.len() - self.color_capacity) / 2
            } else {
                self.color_capacity / 2
            };
            self.slots[slot].locked = decoded.key.cell.lod == 0 && locked < limit;
            self.slots[slot].last = self.frame;
            self.slots[slot].version += 1;
            self.slots[slot].height = decoded.heights;
            let map = decoded.key.map.clone();
            self.resident.insert(decoded.key.clone(), slot);
            self.slots[slot].key = Some(decoded.key);
            if map == "height" {
                self.revision += 1;
            }
            let bytes = if map == "height" {
                self.slots[slot]
                    .height
                    .as_ref()
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect()
            } else {
                decoded.bytes
            };
            return Some((slot, map, bytes));
        }
        None
    }
    pub fn reject(&mut self, slot: usize) {
        if let Some(old) = self.slots[slot].key.take() {
            if old.map == "height" {
                self.revision += 1;
            }
            self.resident.remove(&old);
        }
        self.slots[slot].height = None;
        self.slots[slot].locked = false;
        self.slots[slot].version += 1;
    }
    pub fn reload(&mut self, body: &str, map: Option<&str>) {
        {
            let mut q = self.shared.queue.lock().unwrap();
            q.generation += 1;
            q.jobs.clear();
            q.ready.clear();
            q.inflight.clear();
            q.touched.clear();
        }
        self.catalog = Arc::new(scan(&self.base));
        self.failed.clear();
        let slots: Vec<_> = self
            .resident
            .iter()
            .filter(|(k, _)| k.body == body && map.is_none_or(|m| m == k.map))
            .map(|(_, s)| *s)
            .collect();
        for s in slots {
            self.reject(s);
        }
        self.revision += 1;
    }
    pub fn count(&self) -> usize {
        self.resident.len()
    }
    pub fn versions(&self) -> Vec<u64> {
        self.slots.iter().map(|s| s.version).collect()
    }
    pub fn sample(&mut self, slot: usize, uv: [f64; 2]) -> Option<f64> {
        let width = self.size + GUTTER * 2;
        let s = self.slots.get_mut(slot)?;
        let h = s.height.as_ref()?;
        s.last = self.frame;
        let x = uv[0] * self.size as f64 + GUTTER as f64 - 0.5;
        let y = uv[1] * self.size as f64 + GUTTER as f64 - 0.5;
        let ix = x.floor() as isize;
        let iy = y.floor() as isize;
        let weights = |f: f64| {
            [
                (1. - f).powi(3) / 6.,
                (3. * f.powi(3) - 6. * f * f + 4.) / 6.,
                (-3. * f.powi(3) + 3. * f * f + 3. * f + 1.) / 6.,
                f.powi(3) / 6.,
            ]
        };
        let wx = weights(x - x.floor());
        let wy = weights(y - y.floor());
        let mut value = 0.;
        for j in 0..4 {
            for i in 0..4 {
                let a = (ix + i as isize - 1).clamp(0, width as isize - 1) as usize;
                let b = (iy + j as isize - 1).clamp(0, width as isize - 1) as usize;
                value += h[b * width + a] as f64 * wx[i] * wy[j];
            }
        }
        Some(value)
    }
    pub fn sample_direction(&mut self, body: &str, n: [f64; 3]) -> Option<f64> {
        let (face, uv) = face_uv(n);
        let max = self.max_lod(body, "height").max(0) as u32;
        let mut value = None;
        for lod in 0..=max {
            let count = (1u32 << lod) as f64;
            let p = [
                (uv[0] * count).clamp(0., count),
                (uv[1] * count).clamp(0., count),
            ];
            let xy = [p[0].floor().min(count - 1.), p[1].floor().min(count - 1.)];
            let local = [p[0] - xy[0], p[1] - xy[1]];
            let key = Key {
                body: body.into(),
                map: "height".into(),
                cell: Cell {
                    face,
                    lod,
                    x: xy[0] as u32,
                    y: xy[1] as u32,
                },
            };
            if let Some(&slot) = self.resident.get(&key) {
                if let Some(e) = self.sample(slot, local) {
                    let edge = local[0].min(1. - local[0]).min(local[1]).min(1. - local[1]);
                    let t = (edge * self.size as f64 / 2.).clamp(0., 1.);
                    let w = t * t * (3. - 2. * t);
                    value = Some(value.map(|v| v * (1. - w) + e * w).unwrap_or(e));
                }
            }
        }
        value
    }
    pub fn page_table(&self, bodies: &[(u32, String)]) -> (Vec<u32>, usize) {
        let count = self.resident.keys().filter(|k| k.map == "height").count();
        let capacity = (count * 4).max(64).next_power_of_two();
        let mask = capacity - 1;
        let mut table = vec![u32::MAX; capacity * 4];
        for (body_id, body) in bodies {
            for (key, slot) in &self.resident {
                if &key.body != body || key.map != "height" {
                    continue;
                }
                let code = key.cell.packed();
                let mut at = hash(code ^ body_id.wrapping_mul(0x9e3779b9)) as usize & mask;
                while table[at * 4] != u32::MAX {
                    at = (at + 1) & mask;
                }
                table[at * 4..at * 4 + 4].copy_from_slice(&[
                    code,
                    *body_id,
                    (*slot - self.color_capacity) as u32,
                    0,
                ]);
            }
        }
        (table, mask)
    }
    pub fn stop(&mut self) {
        {
            let mut q = self.shared.queue.lock().unwrap();
            q.closed = true;
            q.jobs.clear();
            q.ready.clear();
            self.shared.wake.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        self.stop();
    }
}
pub fn hash(mut v: u32) -> u32 {
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846ca68b);
    v ^ (v >> 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_cells_unique() {
        let mut codes = HashSet::new();
        for f in 0..6 {
            for lod in 0..5 {
                for x in 0..1 << lod {
                    for y in 0..1 << lod {
                        assert!(codes.insert(Cell { face: f, lod, x, y }.packed()));
                    }
                }
            }
        }
    }
    #[test]
    fn missing_body_does_not_enqueue() {
        let mut s = Stream::new(PathBuf::from("nonexistent-test-terrain"), 16, 8, 1);
        assert_eq!(s.max_lod("missing", "height"), -1);
        assert_eq!(
            s.resolve(Key {
                body: "missing".into(),
                map: "height".into(),
                cell: Cell {
                    face: 4,
                    lod: 6,
                    x: 0,
                    y: 0
                }
            })[0],
            -1.
        );
        assert!(s.shared.queue.lock().unwrap().inflight.is_empty());
    }
}
