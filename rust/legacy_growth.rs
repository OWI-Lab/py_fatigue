use std::f64::consts::PI;

use crate::curves::{Curve, CurveError, CurveKind};

#[derive(Clone, Copy)]
pub enum Geometry<'a> {
    Infinite,
    HollowCylinder {
        diameter: f64,
        thickness: f64,
        width_to_depth: f64,
    },
    Tabulated(FactorTable<'a>),
}

#[derive(Clone, Copy)]
pub struct FactorTable<'a> {
    depths: &'a [f64],
    factors: &'a [f64],
}

impl<'a> FactorTable<'a> {
    pub fn new(depths: &'a [f64], factors: &'a [f64]) -> Result<Self, CurveError> {
        if depths.len() < 2 || depths.len() != factors.len() {
            return Err(CurveError::Shape);
        }
        if depths
            .iter()
            .chain(factors)
            .any(|&x| !x.is_finite() || x <= 0.0)
            || depths.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(CurveError::Range);
        }
        Ok(Self { depths, factors })
    }

    fn reaches_end(self, depth: f64) -> bool {
        depth >= self.depths[self.depths.len() - 1]
    }

    fn factor(self, depth: f64) -> f64 {
        let upper = self
            .depths
            .partition_point(|&sample| sample <= depth)
            .clamp(1, self.depths.len() - 1);
        let lower = upper - 1;
        let fraction = (depth - self.depths[lower]) / (self.depths[upper] - self.depths[lower]);
        self.factors[lower] + fraction.clamp(0.0, 1.0) * (self.factors[upper] - self.factors[lower])
    }
}

fn segment(radius: f64, angle: f64) -> (f64, f64, f64) {
    let area = (angle - (2.0 * angle).sin() / 2.0) * radius.powi(2);
    let centroid = 4.0 * radius * angle.sin().powi(3) / (3.0 * (2.0 * angle - (2.0 * angle).sin()));
    let inertia = radius.powi(4) / 4.0
        * (angle
            - (4.0 * angle).sin() / 4.0
            - 16.0 * angle.sin().powi(6) / (9.0 * (angle - angle.sin() * angle.cos())));
    (area, centroid, inertia)
}

impl Geometry<'_> {
    pub fn factor(self, depth: f64) -> f64 {
        match self {
            Self::Infinite => 1.0,
            Self::Tabulated(table) => table.factor(depth),
            Self::HollowCylinder {
                diameter,
                thickness,
                width_to_depth,
            } => hollow_factor(depth, diameter, thickness, width_to_depth),
        }
    }
}

fn hollow_factor(depth: f64, diameter: f64, thickness: f64, width_to_depth: f64) -> f64 {
    if depth >= thickness {
        return 0.0;
    }
    let outer = diameter / 2.0;
    let inner = outer - thickness;
    let alpha = depth * width_to_depth / outer;
    let vertical = outer * alpha.sin();
    let horizontal = depth - outer * (1.0 - alpha.cos());
    let radius = (vertical.powi(2) + horizontal.powi(2)) / (2.0 * horizontal);
    let beta = (vertical / radius).asin();

    let ring_area = PI * (outer.powi(2) - inner.powi(2));
    let ring_inertia = PI / 4.0 * (outer.powi(4) - inner.powi(4));
    let (area1, offset1, inertia1) = segment(outer, alpha);
    let (area2, offset2, inertia2) = segment(radius, beta);
    let x1 = -offset1;
    let x2 = offset2 - outer * alpha.cos() - radius * beta.cos();
    let axis = -(area1 * x1 + area2 * x2) / (ring_area - area1 - area2);
    let net_inertia = ring_inertia + ring_area * axis.powi(2)
        - (inertia1 + area1 * (axis - x1).powi(2))
        - (inertia2 + area2 * (axis - x2).powi(2));
    // The legacy net-section stress calculation samples this position.
    let stress_location = -(outer * alpha.cos() + axis) + 0.1 * outer * (1.0 + alpha.cos());
    -ring_inertia * stress_location / (outer * net_inertia)
}

#[derive(Debug, PartialEq, Eq)]
pub enum Stop {
    Spectrum,
    Thickness,
    Critical,
    Increment,
    GeometryLimit,
}

pub struct LegacyGrowth<'a> {
    pub stress: &'a [f64],
    pub counts: &'a [f64],
    pub slopes: &'a [f64],
    pub intercepts: &'a [f64],
    pub knees: &'a [f64],
    pub limits: [f64; 2],
    pub geometry: Geometry<'a>,
    pub initial_depth: f64,
}

impl LegacyGrowth<'_> {
    pub fn simulate(&self, output: &mut [[f64; 3]]) -> Result<(usize, Stop), CurveError> {
        if output.len() != self.stress.len() || self.counts.len() != output.len() {
            return Err(CurveError::Shape);
        }
        Curve::new(
            self.slopes,
            self.intercepts,
            self.knees,
            self.limits,
            CurveKind::SifToGrowth,
        )?;
        let [threshold, critical] = self.limits;
        let mut depth = self.initial_depth;
        let table = match self.geometry {
            Geometry::Tabulated(table) => {
                if !depth.is_finite() || depth < table.depths[0] || table.reaches_end(depth) {
                    return Err(CurveError::Range);
                }
                Some(table)
            }
            _ => None,
        };
        for (i, ((&stress, &count), row)) in self
            .stress
            .iter()
            .zip(self.counts)
            .zip(output.iter_mut())
            .enumerate()
        {
            let factor = self.geometry.factor(depth);
            let sif = stress * (PI * depth).sqrt() * factor;
            if factor == 0.0 {
                return Ok((i, Stop::Thickness));
            }
            if sif >= critical {
                return Ok((i, Stop::Critical));
            }
            *row = [depth, sif, factor];
            if sif < threshold {
                continue;
            }
            // The legacy propagation model assigns an exact knee to its lower segment.
            let index = self.knees.partition_point(|&knee| knee < sif);
            let next = depth + self.intercepts[index] * sif.powf(self.slopes[index]) * count;
            if let Some(table) = table {
                if !next.is_finite() || next < depth {
                    return Err(CurveError::Range);
                }
                if table.reaches_end(next) {
                    return Ok((i + 1, Stop::GeometryLimit));
                }
            }
            if next / depth > 10.0 {
                return Ok((i, Stop::Increment));
            }
            depth = next;
        }
        Ok((output.len(), Stop::Spectrum))
    }
}
