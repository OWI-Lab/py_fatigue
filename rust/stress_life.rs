#[derive(Debug, PartialEq, Eq)]
pub enum StressLifeError {
    Shape,
    StaticLoad,
    ZeroDerivative,
}

pub struct Goodman {
    pub amplitude: f64,
    pub ratio_in: f64,
    pub ratio_out: f64,
    pub ultimate: f64,
    pub exponent: f64,
}

impl Goodman {
    pub fn solve(&self, initial: f64) -> Result<f64, StressLifeError> {
        if self.ratio_in == 1.0 {
            return Err(StressLifeError::StaticLoad);
        }
        let term_in =
            (1.0 + self.ratio_in) / (1.0 - self.ratio_in) * self.amplitude / self.ultimate;
        let term_out = (1.0 + self.ratio_out) / (1.0 - self.ratio_out) / self.ultimate;
        let left = (1.0 - term_in.powf(self.exponent)) / self.amplitude;
        let right = term_out.powf(self.exponent);
        let residual =
            |value: f64| value - (left + value.powf(self.exponent - 1.0) * right).recip();
        let mut value = initial;
        let epsilon = 1e-6;
        for _ in 0..1000 {
            let error = residual(value);
            if error.abs() < 1e-6 || error.is_nan() {
                return Ok(if error.is_nan() { f64::NAN } else { value });
            }
            let derivative =
                (residual(value + epsilon) - residual(value - epsilon)) / (2.0 * epsilon);
            if derivative == 0.0 {
                return Err(StressLifeError::ZeroDerivative);
            }
            value -= error / derivative;
        }
        Ok(value)
    }
}

pub fn theil_history(
    stress: &[f64],
    counts: &[f64],
    lives: &[f64],
    sizes: &[usize],
    cycles_out: &mut [f64],
    stress_out: &mut [f64],
) -> Result<(f64, f64), StressLifeError> {
    if [counts.len(), lives.len(), sizes.len()]
        .iter()
        .any(|&len| len != stress.len())
        || cycles_out.len() != stress_out.len()
    {
        return Err(StressLifeError::Shape);
    }
    let mut end = 0.0;
    let mut effective = 0.0;
    let mut offset: usize = 0;
    for index in 0..stress.len() {
        let next = end + counts[index];
        let weight = stress[index] / lives[index];
        let intercept = effective - weight * end;
        let size = sizes[index];
        let end_offset = offset.checked_add(size).ok_or(StressLifeError::Shape)?;
        let cycles = cycles_out
            .get_mut(offset..end_offset)
            .ok_or(StressLifeError::Shape)?;
        let history = stress_out
            .get_mut(offset..end_offset)
            .ok_or(StressLifeError::Shape)?;
        log_segment(
            (end + 1.0).log10(),
            next.log10(),
            weight,
            intercept,
            cycles,
            history,
        );
        offset = end_offset;
        end = next;
        effective = weight * end + intercept;
    }
    Ok((effective, end))
}

fn log_segment(
    left: f64,
    right: f64,
    weight: f64,
    offset: f64,
    cycles: &mut [f64],
    stress: &mut [f64],
) {
    let intervals = cycles.len().saturating_sub(1);
    let step = (right - left) / intervals.max(1) as f64;
    for (index, (cycle, stress)) in cycles.iter_mut().zip(stress).enumerate() {
        let exponent = if index == intervals && intervals > 0 {
            right
        } else {
            left + step * index as f64
        };
        *cycle = 10_f64.powf(exponent);
        *stress = weight * *cycle + offset;
    }
}
