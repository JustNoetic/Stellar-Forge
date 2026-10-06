use crate::{
    quadtree::{direction, dot},
    stream::{hash, Stream},
};
fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0., 1.);
    t * t * (3. - 2. * t)
}
fn noise(p: [f32; 3]) -> f32 {
    let i = p.map(|x| x.floor() as i32);
    let f = std::array::from_fn::<_, 3, _>(|k| {
        let t = p[k] - p[k].floor();
        t * t * t * (t * (t * 6. - 15.) + 10.)
    });
    let mut value = 0.;
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let h = hash(
                    (i[0] + x) as u32
                        ^ ((i[1] + y) as u32).wrapping_mul(0x9e3779b9)
                        ^ ((i[2] + z) as u32).wrapping_mul(0x85ebca6b),
                );
                let n = (h & 0x00ff_ffff) as f32 * (2. / 16777215.) - 1.;
                let w = if x == 0 { 1. - f[0] } else { f[0] };
                let w = w * if y == 0 { 1. - f[1] } else { f[1] };
                let w = w * if z == 0 { 1. - f[2] } else { f[2] };
                value += n * w;
            }
        }
    }
    value
}
pub fn micro(n: [f64; 3], e: f64, lo: f64, span: f64, water: Option<f64>) -> f64 {
    let h = (lo + e * span) as f32;
    let land = water
        .map(|w| smooth(w as f32, w as f32 + 0.04, h))
        .unwrap_or(1.);
    if land <= 0.001 {
        return 0.;
    }
    let mountain = if let Some(w) = water {
        smooth(w as f32 + 0.8, w as f32 + 2.8, h)
    } else {
        smooth(0.25, 0.65, e as f32)
    };
    let mut p = n.map(|v| v as f32 * 450.);
    let mut rolling = 0.;
    let mut ridges = 0.;
    let mut amp = 0.5;
    for _ in 0..4 {
        let v = noise(p);
        rolling += v * amp * 0.45;
        ridges += (1. - v.abs()).powi(2) * amp;
        p = [
            (-0.8 * p[1] - 0.6 * p[2]) * 2.02,
            (0.8 * p[0] + 0.36 * p[1] - 0.48 * p[2]) * 2.02,
            (0.6 * p[0] - 0.48 * p[1] + 0.64 * p[2]) * 2.02,
        ];
        amp *= 0.49;
    }
    ((rolling * (1. - mountain) + (ridges - 0.45) * mountain)
        * (span as f32 * 0.045).min(0.35)
        * land) as f64
}
pub fn vertex(
    stream: &mut Stream,
    body: &str,
    face: u32,
    uv: [f64; 2],
    radius: f64,
    obl: f64,
    lo: f64,
    span: f64,
    water: Option<f64>,
) -> [f64; 3] {
    let n = direction(face, uv[0] * 2. - 1., uv[1] * 2. - 1.);
    let mut elev = 0.;
    if let Some(e) = stream.sample_direction(body, n) {
        elev = lo + e * span;
        if let Some(w) = water {
            elev = elev.max(w);
        }
        elev += micro(n, e, lo, span, water);
    }
    let ell = [n[0], n[1] * (1. - obl), n[2]];
    let len = dot(ell, ell).sqrt();
    std::array::from_fn(|i| ell[i] * radius + ell[i] / len * elev)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detail_is_position_based_and_ocean_mask_is_explicit() {
        let n = [0., 0., 1.];
        let e = micro(n, 0.7, -10., 20., None);
        assert!(e.is_finite());
        assert_eq!(micro(n, 0., -10., 20., Some(0.)), 0.);
        assert_ne!(micro(n, 0.1, -10., 20., None), 0.);
    }
}
