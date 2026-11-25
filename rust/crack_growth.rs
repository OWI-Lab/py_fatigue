use std::f64::consts::PI;

#[derive(Clone, Copy)]
pub enum Plate {
    Center { half_width: f64 },
    Surface { half_width: f64, thickness: f64 },
}

#[derive(Clone, Copy)]
pub struct Load {
    pub minimum: f64,
    pub maximum: f64,
    pub cycles: f64,
}

#[derive(Clone, Copy)]
pub struct Nasgro {
    pub coefficient: f64,
    pub exponent: f64,
    pub p: f64,
    pub q: f64,
    pub threshold_zero: f64,
    pub threshold_fanning: f64,
    pub toughness: f64,
    pub constraint: f64,
    pub peak_to_flow: f64,
}

pub struct Controls {
    pub max_step: f64,
    pub length_tolerance: f64,
    pub cycle_tolerance: f64,
    pub max_trials: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
    pub cycles: f64,
    pub sizes: [f64; 2],
    pub k_min: [f64; 2],
    pub k_max: [f64; 2],
    pub block: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Spectrum,
    Toughness,
    GeometryLimit,
}

#[derive(Debug)]
pub enum GrowthError {
    Geometry,
    Load,
    Material,
    Controls,
    Budget,
    NonFinite,
}

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

impl Plate {
    pub fn factors(self, sizes: [f64; 2]) -> Option<[f64; 2]> {
        let [a, c] = sizes;
        match self {
            Self::Center { half_width }
                if positive(a) && positive(half_width) && a / half_width <= 0.8 =>
            {
                let factor = (PI * a / (PI * a / (2.0 * half_width)).cos()).sqrt();
                Some([factor, 0.0])
            }
            Self::Surface {
                half_width,
                thickness,
            } if positive(half_width)
                && positive(thickness)
                && positive(a)
                && positive(c)
                && a <= c
                && a / thickness <= 0.8
                && c / half_width < 0.5 =>
            {
                Some(surface_factors(a, c, thickness, half_width))
            }
            _ => None,
        }
    }
}

// Newman and Raju, NASA TP-1578, equations 2–10, tension component only.
fn surface_factors(a: f64, c: f64, t: f64, b: f64) -> [f64; 2] {
    let shape = a / c;
    let depth = a / t;
    let q = 1.0 + 1.464 * shape.powf(1.65);
    let m1 = 1.13 - 0.09 * shape;
    let m2 = -0.54 + 0.89 / (0.2 + shape);
    let m3 = 0.5 - 1.0 / (0.65 + shape) + 14.0 * (1.0 - shape).powi(24);
    let width = (PI * c / (2.0 * b) * depth.sqrt()).cos().sqrt().recip();
    let common = (PI * a / q).sqrt() * (m1 + m2 * depth.powi(2) + m3 * depth.powi(4)) * width;
    let surface = common * (1.1 + 0.35 * depth.powi(2)) * shape.sqrt();
    [common, surface]
}

impl Nasgro {
    fn valid(self) -> bool {
        [self.coefficient, self.exponent, self.toughness]
            .iter()
            .copied()
            .all(positive)
            && (1.0..=3.0).contains(&self.constraint)
            && (0.0..1.0).contains(&self.peak_to_flow)
            && [self.p, self.q, self.threshold_zero]
                .iter()
                .all(|&x| x.is_finite() && x >= 0.0)
            && self.threshold_fanning.is_finite()
    }

    fn opening(self, ratio: f64) -> (f64, f64) {
        let alpha = self.constraint;
        let a0 = (0.825 - 0.34 * alpha + 0.05 * alpha.powi(2))
            * (PI * self.peak_to_flow / 2.0).cos().powf(alpha.recip());
        let a1 = (0.415 - 0.071 * alpha) * self.peak_to_flow;
        let a3 = 2.0 * a0 + a1 - 1.0;
        let a2 = 1.0 - a0 - a1 - a3;
        (a0, ratio.max(a0 + ratio * (a1 + ratio * (a2 + ratio * a3))))
    }

    pub fn growth_rate(self, length: f64, k_min: f64, k_max: f64) -> Result<f64, GrowthError> {
        if !self.valid() {
            return Err(GrowthError::Material);
        }
        if !positive(length)
            || !k_min.is_finite()
            || k_min < 0.0
            || !positive(k_max)
            || k_min >= k_max
            || k_max >= self.toughness
        {
            return Err(GrowthError::Load);
        }
        let delta = k_max - k_min;
        let ratio = k_min / k_max;
        let (a0, opening) = self.opening(ratio);
        let closure = (1.0 - opening) / (1.0 - ratio);
        // NASGRO 3.0 uses 0.0015 in (0.0381 mm) as the intrinsic crack length.
        let threshold = self.threshold_zero * (length / (length + 0.0381)).sqrt()
            / (closure / (1.0 - a0)).powf(1.0 + self.threshold_fanning * ratio);
        if self.p > 0.0 && delta <= threshold {
            return Ok(0.0);
        }
        let threshold_term = if self.p == 0.0 {
            1.0
        } else {
            (1.0 - threshold / delta).powf(self.p)
        };
        let rate =
            self.coefficient * ((1.0 - opening) * k_max).powf(self.exponent) * threshold_term
                / (1.0 - k_max / self.toughness).powf(self.q);
        if !rate.is_finite() || rate < 0.0 {
            return Err(GrowthError::NonFinite);
        }
        Ok(rate)
    }
}

struct Evaluation {
    k_min: [f64; 2],
    k_max: [f64; 2],
    rate: [f64; 2],
}

enum State {
    Active(Evaluation),
    Stopped(Stop),
}

fn evaluate(
    plate: Plate,
    material: Nasgro,
    sizes: [f64; 2],
    load: Load,
) -> Result<State, GrowthError> {
    if sizes.iter().any(|value| !value.is_finite()) {
        return Err(GrowthError::NonFinite);
    }
    let Some(factors) = plate.factors(sizes) else {
        return Ok(State::Stopped(Stop::GeometryLimit));
    };
    let k_min = factors.map(|factor| factor * load.minimum);
    let k_max = factors.map(|factor| factor * load.maximum);
    if k_max.iter().any(|k| !k.is_finite()) {
        return Err(GrowthError::NonFinite);
    }
    if k_max.iter().any(|&k| k >= material.toughness) {
        return Ok(State::Stopped(Stop::Toughness));
    }
    let mut rate = [0.0; 2];
    for index in 0..2 {
        if k_max[index] > k_min[index] {
            rate[index] = material.growth_rate(sizes[index], k_min[index], k_max[index])?;
        }
    }
    Ok(State::Active(Evaluation { k_min, k_max, rate }))
}

fn advance(sizes: [f64; 2], rate: [f64; 2], step: f64) -> [f64; 2] {
    [sizes[0] + step * rate[0], sizes[1] + step * rate[1]]
}

fn step_trial(
    plate: Plate,
    material: Nasgro,
    sizes: [f64; 2],
    evaluation: &Evaluation,
    load: Load,
    step: f64,
    length_tolerance: f64,
) -> Result<Trial, GrowthError> {
    let single = advance(sizes, evaluation.rate, step);
    let middle = advance(sizes, evaluation.rate, step / 2.0);
    match evaluate(plate, material, middle, load)? {
        State::Stopped(reason) => Ok(Trial::Stop(reason)),
        State::Active(midpoint) => {
            let double = advance(middle, midpoint.rate, step / 2.0);
            match evaluate(plate, material, double, load)? {
                State::Stopped(reason) => Ok(Trial::Stop(reason)),
                State::Active(next) => {
                    if (single[0] - double[0]).abs() > length_tolerance
                        || (single[1] - double[1]).abs() > length_tolerance
                    {
                        return Ok(Trial::Retry);
                    }
                    Ok(Trial::Accept(double, next))
                }
            }
        }
    }
}

enum Trial {
    Accept([f64; 2], Evaluation),
    Retry,
    Stop(Stop),
}

struct Integration<'a> {
    plate: Plate,
    material: Nasgro,
    control: &'a Controls,
    output: &'a mut [Point],
    written: usize,
    trials: usize,
    sizes: [f64; 2],
    elapsed: f64,
}

impl Integration<'_> {
    fn record(&mut self, evaluation: &Evaluation, block: usize) -> Result<(), GrowthError> {
        let slot = self
            .output
            .get_mut(self.written)
            .ok_or(GrowthError::Budget)?;
        *slot = Point {
            cycles: self.elapsed,
            sizes: self.sizes,
            k_min: evaluation.k_min,
            k_max: evaluation.k_max,
            block,
        };
        self.written += 1;
        Ok(())
    }

    fn integrate(&mut self, index: usize, load: Load) -> Result<Option<Stop>, GrowthError> {
        let mut evaluation = match evaluate(self.plate, self.material, self.sizes, load)? {
            State::Active(evaluation) => evaluation,
            State::Stopped(reason) => return Ok(Some(reason)),
        };
        self.record(&evaluation, index)?;
        let mut remaining = load.cycles;
        let mut step = self.control.max_step.min(remaining);
        for _ in self.trials..self.control.max_trials {
            if remaining == 0.0 {
                return Ok(None);
            }
            if !positive(step) || self.elapsed + step == self.elapsed {
                return Err(GrowthError::Budget);
            }
            self.trials += 1;
            match step_trial(
                self.plate,
                self.material,
                self.sizes,
                &evaluation,
                load,
                step,
                self.control.length_tolerance,
            )? {
                Trial::Stop(reason) if step <= self.control.cycle_tolerance => {
                    return Ok(Some(reason));
                }
                Trial::Retry | Trial::Stop(_) => step /= 2.0,
                Trial::Accept(sizes, next) => {
                    self.sizes = sizes;
                    self.elapsed += step;
                    if !self.elapsed.is_finite() {
                        return Err(GrowthError::NonFinite);
                    }
                    remaining -= step;
                    evaluation = next;
                    self.record(&evaluation, index)?;
                    step = (step * 2.0).min(self.control.max_step).min(remaining);
                }
            }
        }
        if remaining == 0.0 {
            Ok(None)
        } else {
            Err(GrowthError::Budget)
        }
    }
}

pub fn simulate(
    plate: Plate,
    initial: [f64; 2],
    material: Nasgro,
    loads: &[Load],
    control: &Controls,
    output: &mut [Point],
) -> Result<(usize, Stop), GrowthError> {
    if plate.factors(initial).is_none() {
        return Err(GrowthError::Geometry);
    }
    if !material.valid() {
        return Err(GrowthError::Material);
    }
    if loads.is_empty()
        || loads.iter().any(|load| {
            !load.minimum.is_finite()
                || load.minimum < 0.0
                || !positive(load.maximum)
                || load.minimum > load.maximum
                || !positive(load.cycles)
        })
    {
        return Err(GrowthError::Load);
    }
    if !positive(control.max_step)
        || !positive(control.length_tolerance)
        || !positive(control.cycle_tolerance)
        || control.max_trials == 0
    {
        return Err(GrowthError::Controls);
    }
    let mut integration = Integration {
        plate,
        material,
        control,
        output,
        written: 0,
        trials: 0,
        sizes: initial,
        elapsed: 0.0,
    };
    for (index, &load) in loads.iter().enumerate() {
        if let Some(reason) = integration.integrate(index, load)? {
            return Ok((integration.written, reason));
        }
    }
    Ok((integration.written, Stop::Spectrum))
}
