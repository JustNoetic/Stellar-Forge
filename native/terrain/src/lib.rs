mod quadtree;
mod stream;
mod surface;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use quadtree::{Cell, View};
use std::path::PathBuf;
use stream::{Key, Stream};

fn f32s(data: &[u8]) -> PyResult<Vec<f32>> {
    if data.len() % 4 != 0 {
        return Err(PyValueError::new_err("unaligned float32 buffer"));
    }
    Ok(data
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn f64s(data: &[u8]) -> PyResult<Vec<f64>> {
    if data.len() % 8 != 0 {
        return Err(PyValueError::new_err("unaligned float64 buffer"));
    }
    Ok(data
        .chunks_exact(8)
        .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn bytes32<'py>(py: Python<'py>, values: &[f32]) -> Bound<'py, PyBytes> {
    PyBytes::new(
        py,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn cell(face: u32, lod: u32, x: u32, y: u32) -> PyResult<Cell> {
    if face > 5 || lod > 20 || x >= 1 << lod || y >= 1 << lod {
        return Err(PyValueError::new_err("invalid terrain cell"));
    }
    Ok(Cell { face, lod, x, y })
}

#[pyfunction]
#[pyo3(signature=(camera,radius,obl,pixels,tan_fov,threshold,max_lod,budget,planes,cloud,bend,height,previous=None,altitude=None))]
#[allow(clippy::too_many_arguments)]
fn traverse<'py>(
    py: Python<'py>,
    camera: [f64; 3],
    radius: f64,
    obl: f64,
    pixels: f64,
    tan_fov: f64,
    threshold: f64,
    max_lod: u32,
    budget: usize,
    planes: Vec<[f64; 4]>,
    cloud: f64,
    bend: f64,
    height: f64,
    previous: Option<&Bound<'py, PyBytes>>,
    altitude: Option<f64>,
) -> PyResult<Bound<'py, PyBytes>> {
    if !camera
        .iter()
        .chain([radius, obl, pixels, tan_fov, threshold, cloud, bend, height].iter())
        .all(|v| v.is_finite())
        || radius <= 0.
        || !(0.0..0.99).contains(&obl)
        || tan_fov <= 0.
        || max_lod > 20
        || budget > 65536
        || planes.iter().flatten().any(|v| !v.is_finite())
        || altitude.is_some_and(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err("invalid terrain view"));
    }
    let view = View {
        camera,
        radius,
        obl,
        pixels,
        tan_fov,
        threshold,
        max_lod,
        budget,
        planes,
        cloud,
        bend,
        height,
        altitude,
    };
    let previous = match previous {
        Some(b) => patch_cells(b.as_bytes())?,
        None => vec![],
    };
    let values = py.detach(|| {
        if previous.is_empty() {
            quadtree::traverse(&view)
        } else {
            quadtree::traverse_with_history(&view, &previous)
        }
    });
    Ok(bytes32(py, &values))
}

fn patch_cells(data: &[u8]) -> PyResult<Vec<Cell>> {
    let values = f32s(data)?;
    if values.len() % 10 != 0 || values.len() > 655360 || values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("invalid patch history"));
    }
    values
        .chunks_exact(10)
        .map(|p| cell(p[4] as u32, p[5] as u32, p[6] as u32, p[7] as u32))
        .collect()
}

#[pyfunction]
#[pyo3(signature=(patches,camera,radius,obl,extra,planes,bend=0.,height_range=None))]
fn cull<'py>(
    py: Python<'py>,
    patches: &Bound<'py, PyBytes>,
    camera: [f64; 3],
    radius: f64,
    obl: f64,
    extra: f64,
    planes: Vec<[f64; 4]>,
    bend: f64,
    height_range: Option<[f64; 2]>,
) -> PyResult<Bound<'py, PyBytes>> {
    let cells = patch_cells(patches.as_bytes())?;
    let [minimum, maximum] = height_range.unwrap_or([-extra.max(0.), extra.max(0.)]);
    if !camera
        .iter()
        .chain([radius, obl, extra, bend].iter())
        .chain(planes.iter().flatten())
        .all(|v| v.is_finite())
        || radius <= 0.
        || !(0.0..0.99).contains(&obl)
        || !minimum.is_finite()
        || !maximum.is_finite()
        || minimum > maximum
    {
        return Err(PyValueError::new_err("invalid culling view"));
    }
    let rows = f32s(patches.as_bytes())?;
    let view = View {
        camera,
        radius,
        obl,
        pixels: 1.,
        tan_fov: 1.,
        threshold: 1.,
        max_lod: 20,
        budget: 65536,
        planes,
        cloud: 0.,
        bend,
        height: extra.max(minimum.abs()).max(maximum.abs()),
        altitude: None,
    };
    let values = py.detach(|| {
        let mut out = vec![];
        for (c, p) in cells.into_iter().zip(rows.chunks_exact(10)) {
            if view.visible_with_heights(c, p[8] as f64, minimum, maximum) {
                out.extend_from_slice(p);
            }
        }
        out
    });
    Ok(bytes32(py, &values))
}
#[pyfunction]
fn pack<'py>(
    py: Python<'py>,
    raw: &Bound<'py, PyBytes>,
    layers: &Bound<'py, PyBytes>,
    body: f32,
    lo: f32,
    span: f32,
    caster: f32,
    body_slot: f32,
    water: Option<f32>,
    cloud: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let r = f32s(raw.as_bytes())?;
    let l = f32s(layers.as_bytes())?;
    let n = r.len() / 10;
    if r.len() % 10 != 0 || l.len() != n * 8 {
        return Err(PyValueError::new_err("patch/layer buffer shape mismatch"));
    }
    let result = py.detach(|| {
        let mut out = vec![0f32; n * 24];
        for i in 0..n {
            let p = &r[i * 10..i * 10 + 10];
            let a = &l[i * 8..i * 8 + 8];
            let o = &mut out[i * 24..i * 24 + 24];
            o[..4].copy_from_slice(&p[..4]);
            o[4..8].copy_from_slice(&[a[1], a[2], a[3], if cloud { 0. } else { p[8] * 0.05 }]);
            o[8..12].copy_from_slice(&[p[4], p[5], a[0], body]);
            o[12..16].copy_from_slice(&a[4..8]);
            o[16..20].copy_from_slice(&[lo, span, caster, body_slot]);
            o[20] = if cloud { 0. } else { p[9] };
            o[21] = if water.is_some() { 1. } else { 0. };
            o[22] = water.unwrap_or(0.);
        }
        out
    });
    Ok(bytes32(py, &result))
}
#[pyclass]
struct TileBackend {
    inner: Stream,
}
#[pymethods]
impl TileBackend {
    #[new]
    fn new(
        py: Python<'_>,
        base: String,
        capacity: usize,
        size: usize,
        workers: usize,
    ) -> PyResult<Self> {
        if !(16..=4096).contains(&capacity)
            || !(8..=1024).contains(&size)
            || !(1..=16).contains(&workers)
        {
            return Err(PyValueError::new_err("invalid terrain pool size"));
        }
        Ok(Self {
            inner: py.detach(|| Stream::new(PathBuf::from(base), capacity, size, workers)),
        })
    }
    #[getter]
    fn color_capacity(&self) -> usize {
        self.inner.color_capacity
    }
    #[getter]
    fn revision(&self) -> u64 {
        self.inner.revision
    }
    #[getter]
    fn resident_count(&self) -> usize {
        self.inner.count()
    }
    #[getter]
    fn versions(&self) -> Vec<u64> {
        self.inner.versions()
    }
    fn max_lod(&self, body: &str, map: &str) -> i32 {
        self.inner.max_lod(body, map)
    }
    fn begin_frame(&mut self) {
        self.inner.begin_frame();
    }
    #[pyo3(signature=(body,map,face,lod,x,y,priority=None))]
    fn request(
        &mut self,
        body: String,
        map: String,
        face: u32,
        lod: u32,
        x: u32,
        y: u32,
        priority: Option<u32>,
    ) -> PyResult<()> {
        self.inner.request(
            Key {
                body: body.to_lowercase(),
                map,
                cell: cell(face, lod, x, y)?,
            },
            priority,
        );
        Ok(())
    }
    fn resolve(
        &mut self,
        body: String,
        map: String,
        face: u32,
        lod: u32,
        x: u32,
        y: u32,
    ) -> PyResult<[f32; 4]> {
        Ok(self.inner.resolve(Key {
            body: body.to_lowercase(),
            map,
            cell: cell(face, lod, x, y)?,
        }))
    }
    fn is_resident(
        &self,
        body: String,
        map: String,
        face: u32,
        lod: u32,
        x: u32,
        y: u32,
    ) -> PyResult<bool> {
        Ok(self.inner.is_resident(Key {
            body: body.to_lowercase(),
            map,
            cell: cell(face, lod, x, y)?,
        }))
    }
    fn resolve_batch<'py>(
        &mut self,
        py: Python<'py>,
        body: String,
        maps: Vec<String>,
        coords: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let coords = f32s(coords.as_bytes())?;
        if coords.len() % 4 != 0 {
            return Err(PyValueError::new_err("invalid tile coordinate buffer"));
        }
        let cells: Vec<_> = coords
            .chunks_exact(4)
            .map(|p| cell(p[0] as u32, p[1] as u32, p[2] as u32, p[3] as u32))
            .collect::<PyResult<_>>()?;
        let body = body.to_lowercase();
        let values = py.detach(|| {
            let mut out = Vec::with_capacity(cells.len() * maps.len() * 4);
            for map in maps {
                for &cell in &cells {
                    out.extend(self.inner.resolve(Key {
                        body: body.clone(),
                        map: map.clone(),
                        cell,
                    }));
                }
            }
            out
        });
        Ok(bytes32(py, &values))
    }
    fn poll<'py>(&mut self, py: Python<'py>) -> Option<(usize, String, Bound<'py, PyBytes>)> {
        self.inner
            .poll()
            .map(|(s, m, b)| (s, m, PyBytes::new(py, &b)))
    }
    fn reject(&mut self, slot: usize) -> PyResult<()> {
        if slot >= self.inner.versions().len() {
            return Err(PyValueError::new_err("invalid terrain slot"));
        }
        self.inner.reject(slot);
        Ok(())
    }
    #[pyo3(signature=(body,map=None))]
    fn reload(&mut self, py: Python<'_>, body: String, map: Option<String>) {
        py.detach(|| self.inner.reload(&body.to_lowercase(), map.as_deref()));
    }
    fn sample_height<'py>(
        &mut self,
        py: Python<'py>,
        slot: i32,
        uv: &Bound<'py, PyBytes>,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        if slot < 0 {
            return Ok(None);
        }
        let coords = f64s(uv.as_bytes())?;
        if coords.len() % 2 != 0 {
            return Err(PyValueError::new_err("invalid height UV buffer"));
        }
        let values: Option<Vec<_>> = coords
            .chunks_exact(2)
            .map(|v| self.inner.sample(slot as usize, [v[0], v[1]]))
            .collect();
        Ok(values.map(|v| {
            PyBytes::new(
                py,
                &v.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>(),
            )
        }))
    }
    fn surface_vertices<'py>(
        &mut self,
        py: Python<'py>,
        body: String,
        face: u32,
        lod: u32,
        x: u32,
        y: u32,
        grid: &Bound<'py, PyBytes>,
        radius: f64,
        obl: f64,
        lo: f64,
        span: f64,
        water: Option<f64>,
        edges: u32,
        resolution: u32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let c = cell(face, lod, x, y)?;
        let grid = f64s(grid.as_bytes())?;
        if grid.len() % 2 != 0 || resolution == 0 {
            return Err(PyValueError::new_err("invalid surface grid"));
        }
        let body = body.to_lowercase();
        let values = py.detach(|| {
            let mut out = Vec::with_capacity(grid.len() / 2 * 3);
            let count = (1u32 << c.lod) as f64;
            for p in grid.chunks_exact(2) {
                let point = |s: &mut Stream, g: [f64; 2]| {
                    surface::vertex(
                        s,
                        &body,
                        face,
                        [(x as f64 + g[0]) / count, (y as f64 + g[1]) / count],
                        radius,
                        obl,
                        lo,
                        span,
                        water,
                    )
                };
                let mut v = point(&mut self.inner, [p[0], p[1]]);
                let ix = (p[0] * resolution as f64).round() as u32;
                let iy = (p[1] * resolution as f64).round() as u32;
                let axis = if (ix == 0 && edges & 1 != 0 || ix == resolution && edges & 2 != 0)
                    && iy % 2 == 1
                {
                    Some(1)
                } else if (iy == 0 && edges & 4 != 0 || iy == resolution && edges & 8 != 0)
                    && ix % 2 == 1
                {
                    Some(0)
                } else {
                    None
                };
                if let Some(axis) = axis {
                    let mut a = [p[0], p[1]];
                    let mut b = a;
                    a[axis] -= 1. / resolution as f64;
                    b[axis] += 1. / resolution as f64;
                    let va = point(&mut self.inner, a);
                    let vb = point(&mut self.inner, b);
                    v = std::array::from_fn(|i| (va[i] + vb[i]) * 0.5);
                }
                out.extend(v);
            }
            out
        });
        Ok(PyBytes::new(
            py,
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        ))
    }
    fn surface_altitude(
        &mut self,
        py: Python<'_>,
        body: String,
        camera: [f64; 3],
        radius: f64,
        obl: f64,
        lo: f64,
        span: f64,
        water: Option<f64>,
    ) -> PyResult<f64> {
        if !camera
            .iter()
            .chain([radius, obl, lo, span].iter())
            .all(|v| v.is_finite())
            || radius <= 0.
            || !(0.0..0.99).contains(&obl)
            || water.is_some_and(|v| !v.is_finite())
        {
            return Err(PyValueError::new_err("invalid surface altitude query"));
        }
        let body = body.to_lowercase();
        Ok(py.detach(|| {
            let shaped = [camera[0], camera[1] / (1. - obl), camera[2]];
            if quadtree::dot(shaped, shaped) < 1e-18 {
                return -radius;
            }
            let (face, uv) = quadtree::face_uv(shaped);
            let surface = surface::vertex(
                &mut self.inner,
                &body,
                face,
                uv,
                radius,
                obl,
                lo,
                span,
                water,
            );
            quadtree::dot(camera, camera).sqrt() - quadtree::dot(surface, surface).sqrt()
        }))
    }
    fn height_table<'py>(
        &self,
        py: Python<'py>,
        bodies: Vec<(u32, String)>,
    ) -> (Bound<'py, PyBytes>, usize) {
        let bodies: Vec<_> = bodies
            .into_iter()
            .map(|(i, n)| (i, n.to_lowercase()))
            .collect();
        let (values, mask) = self.inner.page_table(&bodies);
        (
            PyBytes::new(
                py,
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            mask,
        )
    }
    fn errors(&mut self) -> Vec<String> {
        self.inner.errors.drain(..).collect()
    }
    fn shutdown(&mut self, py: Python<'_>) {
        py.detach(|| self.inner.stop());
    }
}
#[pymodule]
fn stellar_terrain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(traverse, m)?)?;
    m.add_function(wrap_pyfunction!(cull, m)?)?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
    m.add_class::<TileBackend>()?;
    m.add("PATCH_FLOATS", 24)?;
    m.add("GUTTER", stream::GUTTER)?;
    Ok(())
}
