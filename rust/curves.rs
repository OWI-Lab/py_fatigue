#[derive(Debug, PartialEq, Eq)]
pub enum CurveError {
    Shape,
    Range,
}

#[derive(Clone, Copy)]
pub enum CurveKind {
    StressToCycles,
    CyclesToStress,
    SifToGrowth,
    GrowthToSif,
}

pub struct Curve<'a> {
    slopes: &'a [f64],
    intercepts: &'a [f64],
    knees: &'a [f64],
    limits: [f64; 2],
    kind: CurveKind,
}

impl<'a> Curve<'a> {
    pub fn new(
        slopes: &'a [f64],
        intercepts: &'a [f64],
        knees: &'a [f64],
        limits: [f64; 2],
        kind: CurveKind,
    ) -> Result<Self, CurveError> {
        if slopes.is_empty() || slopes.len() != intercepts.len() {
            return Err(CurveError::Shape);
        }
        let expected = match kind {
            CurveKind::StressToCycles | CurveKind::CyclesToStress => slopes.len() + 1,
            CurveKind::SifToGrowth | CurveKind::GrowthToSif => slopes.len() - 1,
        };
        if knees.len() != expected {
            return Err(CurveError::Shape);
        }
        Ok(Self {
            slopes,
            intercepts,
            knees,
            limits,
            kind,
        })
    }

    pub fn evaluate(&self, input: f64) -> Result<f64, CurveError> {
        match self.kind {
            CurveKind::StressToCycles if input >= 0.0 => Ok(self.stress_to_cycles(input)),
            CurveKind::CyclesToStress if input > 0.0 => Ok(self.cycles_to_stress(input)),
            CurveKind::SifToGrowth if input >= 0.0 => Ok(self.sif_to_growth(input)),
            CurveKind::GrowthToSif if input >= 0.0 => Ok(self.growth_to_sif(input)),
            _ => Err(CurveError::Range),
        }
    }

    pub fn fill(&self, values: &[f64], output: &mut [f64]) -> Result<(), CurveError> {
        if values.is_empty() || values.len() != output.len() {
            return Err(CurveError::Shape);
        }
        for (&value, slot) in values.iter().zip(output) {
            *slot = self.evaluate(value)?;
        }
        Ok(())
    }

    fn sn_bin(&self, value: f64) -> usize {
        if self.knees[0] <= self.knees[self.knees.len() - 1] {
            self.knees.partition_point(|&knee| knee <= value)
        } else {
            self.knees.partition_point(|&knee| knee > value)
        }
    }

    fn stress_to_cycles(&self, input: f64) -> f64 {
        let log_stress = input.log10();
        let index = self
            .sn_bin(log_stress)
            .saturating_sub(1)
            .min(self.slopes.len() - 1);
        let cycles = 10_f64.powf(self.intercepts[index] - self.slopes[index] * log_stress);
        if cycles > self.limits[0] {
            f64::INFINITY
        } else {
            cycles
        }
    }

    fn cycles_to_stress(&self, input: f64) -> f64 {
        let log_cycles = input.log10();
        let index = self.sn_bin(log_cycles).saturating_sub(1);
        let logarithm = if index == self.slopes.len() {
            (self.limits[0].log10() - log_cycles) / self.slopes[index - 1]
        } else {
            (self.intercepts[index] - log_cycles) / self.slopes[index]
        };
        let stress = 10_f64.powf(logarithm);
        if stress < self.limits[1] {
            self.limits[1]
        } else {
            stress
        }
    }

    fn sif_to_growth(&self, sif: f64) -> f64 {
        let [threshold, critical] = self.limits;
        if sif < threshold {
            return 0.0;
        }
        if sif > critical || sif == f64::INFINITY {
            return f64::INFINITY;
        }
        let index = self.knees.partition_point(|&knee| knee <= sif);
        self.intercepts[index] * sif.powf(self.slopes[index])
    }

    fn growth_to_sif(&self, rate: f64) -> f64 {
        let [threshold, critical] = self.limits;
        let last = self.slopes.len() - 1;
        let lower = self.intercepts[0] * (0.9999999999999999 * threshold).powf(self.slopes[0]);
        let upper = self.intercepts[last] * (critical / 0.9999999999999999).powf(self.slopes[last]);
        if rate < lower {
            return threshold;
        }
        if rate >= upper {
            return critical;
        }
        let index = self.knees.partition_point(|&knee| knee <= rate);
        (rate / self.intercepts[index]).powf(1.0 / self.slopes[index])
    }
}

pub fn sn_intersection(
    curve: &Curve<'_>,
    stress: f64,
    weight: f64,
    mut left: f64,
    mut right: f64,
    tol: f64,
    iterations: usize,
) -> Result<f64, CurveError> {
    let residual = |cycles| Ok(curve.evaluate(cycles)? - weight * cycles - stress);
    let mut fa: f64 = residual(left)?;
    if fa.abs() < tol {
        return Ok(left);
    }
    let fb: f64 = residual(right)?;
    if fb.abs() < tol {
        return Ok(right);
    }
    let mut midpoint = (left + right) / 2.0;
    for _ in 0..iterations {
        let fm: f64 = residual(midpoint)?;
        if fm.abs() <= tol || fm.is_nan() {
            break;
        }
        if fa * fm < 0.0 {
            right = midpoint;
        } else {
            left = midpoint;
            fa = fm;
        }
        midpoint = (left + right) / 2.0;
    }
    Ok(midpoint)
}
