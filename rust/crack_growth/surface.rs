use std::f64::consts::{FRAC_PI_2, PI};

use super::positive;

// NASA TM-87594, Table 1. Axes are a/c, R_inner/t, a/t, and tip (A, B).
const PIPE_TENSION: [[[[f64; 2]; 3]; 4]; 3] = [
    [
        [[1.113, 0.943], [1.226, 1.080], [1.455, 1.327]],
        [[1.105, 0.937], [1.194, 1.070], [1.342, 1.285]],
        [[1.101, 0.933], [1.178, 1.071], [1.285, 1.285]],
        [[1.097, 0.930], [1.167, 1.070], [1.247, 1.290]],
    ],
    [
        [[1.061, 1.060], [1.114, 1.161], [1.202, 1.354]],
        [[1.059, 1.056], [1.106, 1.155], [1.174, 1.331]],
        [[1.058, 1.053], [1.103, 1.156], [1.157, 1.333]],
        [[1.057, 1.051], [1.101, 1.156], [1.144, 1.335]],
    ],
    [
        [[1.015, 1.160], [1.036, 1.237], [1.076, 1.385]],
        [[1.017, 1.157], [1.041, 1.235], [1.072, 1.383]],
        [[1.019, 1.154], [1.046, 1.234], [1.072, 1.381]],
        [[1.020, 1.152], [1.049, 1.233], [1.074, 1.380]],
    ],
];

fn interpolation_cell(axis: &[f64], value: f64) -> Option<(usize, f64)> {
    axis.windows(2).enumerate().find_map(|(index, cell)| {
        let lo = cell[0];
        let hi = cell[1];
        (lo..=hi)
            .contains(&value)
            .then(|| (index, (value - lo) / (hi - lo)))
    })
}

fn interpolate(lower: f64, upper: f64, fraction: f64) -> f64 {
    lower + fraction * (upper - lower)
}

fn ellipse_shape(ratio: f64) -> f64 {
    1.0 + 1.464 * ratio.powf(1.65)
}

pub(super) fn external_pipe_factors(a: f64, c: f64, r: f64, t: f64) -> Option<[f64; 2]> {
    if ![a, c, r, t, r + t].into_iter().all(positive) {
        return None;
    }
    let ratio = a / c;
    let (shape, shape_weight) = interpolation_cell(&[0.6, 0.8, 1.0], ratio)?;
    let (radius, radius_weight) = interpolation_cell(&[1.0, 2.0, 4.0, 10.0], r / t)?;
    let (depth, depth_weight) = interpolation_cell(&[0.2, 0.5, 0.8], a / t)?;
    let scale = (PI / ellipse_shape(ratio)).sqrt() * a.sqrt();
    let factor = |tip| {
        let by_depth = |shape_index: usize, radius_index: usize| {
            let values = &PIPE_TENSION[shape_index][radius_index];
            interpolate(values[depth][tip], values[depth + 1][tip], depth_weight)
        };
        let by_radius = |shape_index| {
            interpolate(
                by_depth(shape_index, radius),
                by_depth(shape_index, radius + 1),
                radius_weight,
            )
        };
        scale * interpolate(by_radius(shape), by_radius(shape + 1), shape_weight)
    };
    Some([factor(0), factor(1)])
}

// Mettu et al., NLR-TP-99313, Section 5, Table 2 (a/c = 1) and Fig. 2 (a/d).
pub(super) fn rolled_thread_factor(a: f64, minor: f64, major: f64) -> Option<f64> {
    if !positive(a) || !positive(minor) || !positive(major) || minor >= major {
        return None;
    }
    let factors = [1.00, 0.84, 0.76, 0.65, 0.59, 0.62, 1.00];
    let (index, weight) = interpolation_cell(&[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], a / minor)?;
    Some(PI.sqrt() * a.sqrt() * interpolate(factors[index], factors[index + 1], weight))
}

// NASA TM-85793, equations 53–64. Here a is the bore half-span and c is
// the radial depth. Return the radial tip first, then the bore tip.
pub(super) fn bore_factors(a: f64, c: f64, t: f64, b: f64, r: f64, count: f64) -> Option<[f64; 2]> {
    if ![a, c, t, b, r].into_iter().all(positive) || (count != 1.0 && count != 2.0) {
        return None;
    }
    let shape = a / c;
    let span = a / t;
    if !(0.2..=2.0).contains(&shape)
        || span >= 1.0
        || !(0.5..=2.0).contains(&(r / t))
        || r / b + c / b >= 0.5
    {
        return None;
    }
    let width_angle = PI * (2.0 * (r / b) + count * (c / b)) * span.sqrt()
        / (4.0 + (2.0 * count - 4.0) * (c / b));
    let width = ((FRAC_PI_2 * (r / b)).cos() * width_angle.cos())
        .sqrt()
        .recip();
    let area_ratio = span * (c / r);
    let single = if count == 1.0 {
        ((4.0 / PI + area_ratio / 2.0) / (4.0 / PI + area_ratio)).sqrt()
    } else {
        1.0
    };
    let ratio = shape.min(shape.recip());
    let m1 = shape.max(1.0).sqrt().recip();
    let m2 = 0.05 / (0.11 + shape.powf(1.5));
    let m3 = 0.29 / (0.23 + shape.powf(1.5));
    let base = (PI / ellipse_shape(ratio)).sqrt()
        * a.sqrt()
        * (m1 + m2 * span.powi(2) + m3 * span.powi(4))
        * single
        * width;
    let radial = 1.0 - span.powi(4) * (2.6 - 2.0 * span).sqrt() / (1.0 + 4.0 * shape);
    let bore = 1.0 + 0.1 * (1.0 - span).powi(10);
    let radial_shape = shape.min(1.0).sqrt();
    let bore_shape = shape.recip().min(1.0).sqrt();
    Some([
        base * radial * radial_shape * hole_factor(1.0 / (1.0 + c / r)),
        base * bore * bore_shape * hole_factor(1.0 / (1.0 + (c / r) * (0.9 * FRAC_PI_2).cos())),
    ])
}

fn hole_factor(lambda: f64) -> f64 {
    let squared = lambda.powi(2);
    (1.0 + 0.358 * lambda + squared * (1.425 - 1.578 * lambda + 2.156 * squared))
        / (1.0 + 0.08 * squared)
}
