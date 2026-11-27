use std::f64::consts::PI;

mod surface;
use surface::{bore_factors, external_pipe_factors, rolled_thread_factor};

#[derive(Clone, Copy)]
pub enum Plate {
    Center {
        half_width: f64,
    },
    Edge {
        width: f64,
    },
    Surface {
        half_width: f64,
        thickness: f64,
    },
    SurfaceRoundBar {
        diameter: f64,
    },
    SurfaceExternalPipe {
        inner_radius: f64,
        thickness: f64,
    },
    SurfaceRolledThread {
        minor: f64,
        major: f64,
    },
    SurfaceHoleBore {
        width: f64,
        thickness: f64,
        hole_diameter: f64,
        count: f64,
    },
    SphereThrough {
        radius: f64,
        thickness: f64,
    },
    AxialCylinderThrough {
        radius: f64,
        thickness: f64,
    },
    CircumferentialCylinderThrough {
        radius: f64,
        thickness: f64,
        length: f64,
        poisson: f64,
    },
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
    HistoryCapacity,
    TrialBudget {
        block: usize,
        cycles: f64,
        limit: usize,
    },
    CycleResolution {
        block: usize,
        cycles: f64,
    },
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
            Self::Edge { width } if positive(a) && positive(width) && a / width <= 0.8 => {
                Some([edge_factor(a, width), 0.0])
            }
            Self::Surface {
                half_width,
                thickness,
            } if positive(half_width)
                && positive(thickness)
                && positive(a)
                && positive(c)
                && a / c <= 1.2
                && a / thickness <= 0.8
                && c / half_width < 0.5 =>
            {
                Some(surface_factors(a, c, thickness, half_width))
            }
            Self::SurfaceRoundBar { diameter }
                if positive(a) && positive(diameter) && a < diameter / 2.0 =>
            {
                Some([round_bar_tension_factor(a, diameter), 0.0])
            }
            Self::SurfaceExternalPipe {
                inner_radius,
                thickness,
            } => external_pipe_factors(a, c, inner_radius, thickness),
            Self::SurfaceRolledThread { minor, major } => {
                Some([rolled_thread_factor(a, minor, major)?, 0.0])
            }
            Self::SurfaceHoleBore {
                width,
                thickness,
                hole_diameter,
                count,
            } => bore_factors(
                c,
                a,
                thickness / 2.0,
                width / 2.0,
                hole_diameter / 2.0,
                count,
            ),
            Self::SphereThrough { radius, thickness } => {
                let lambda = shell_parameter(a, radius, thickness)?;
                (lambda <= 3.0 && a / radius < PI).then(|| {
                    [
                        PI.sqrt() * a.sqrt() * (1.0 + 3.0 * lambda.powf(1.9)).powf(0.4),
                        0.0,
                    ]
                })
            }
            Self::AxialCylinderThrough { radius, thickness } => {
                let lambda = shell_parameter(a, radius, thickness)?;
                (lambda <= 10.0).then(|| {
                    let bulging =
                        (1.0 + 0.52 * lambda + (1.29 - 0.074 * lambda) * lambda.powi(2)).sqrt();
                    [PI.sqrt() * a.sqrt() * bulging, 0.0]
                })
            }
            Self::CircumferentialCylinderThrough {
                radius,
                thickness,
                length,
                poisson,
            } => {
                let factor = circumferential_tension_factor(a, radius, thickness, length, poisson)?;
                Some([factor, 0.0])
            }
            _ => None,
        }
    }

    pub fn net_section_stress(self, sizes: [f64; 2], stress: f64) -> Result<f64, GrowthError> {
        if !stress.is_finite() || stress < 0.0 {
            return Err(GrowthError::Load);
        }
        if self.factors(sizes).is_none() {
            return Err(GrowthError::Geometry);
        }
        let [a, c] = sizes;
        let factor = match self {
            Self::Center { half_width } => 1.0 / (1.0 - a / half_width),
            Self::Edge { width } => {
                let crack = a / width;
                let ligament = 1.0 - crack;
                1.0 / ligament + 3.0 * crack * (1.0 - 2.0 * 0.13) / ligament.powi(2)
            }
            Self::Surface {
                half_width,
                thickness,
            } => surface_net_section_factor(a / thickness, c / half_width),
            Self::SurfaceRoundBar { .. }
            | Self::SurfaceExternalPipe { .. }
            | Self::SurfaceRolledThread { .. }
            | Self::SurfaceHoleBore { .. }
            | Self::SphereThrough { .. }
            | Self::AxialCylinderThrough { .. }
            | Self::CircumferentialCylinderThrough { .. } => return Err(GrowthError::Geometry),
        };
        let value = stress * factor;
        if !value.is_finite() || !positive(factor) {
            return Err(GrowthError::NonFinite);
        }
        Ok(value)
    }
}

// NASA CR-4706, pp. A-27 and A-36. Use the mid-wall radius.
fn shell_parameter(a: f64, radius: f64, thickness: f64) -> Option<f64> {
    if !positive(a) || !positive(radius) || !positive(thickness) || thickness / radius >= 2.0 {
        return None;
    }
    Some(a / (radius.sqrt() * thickness.sqrt()))
}

// Forman et al., DTD Handbook Table 11.3.13, axial tension component.
fn circumferential_tension_factor(
    a: f64,
    radius: f64,
    thickness: f64,
    length: f64,
    poisson: f64,
) -> Option<f64> {
    shell_parameter(a, radius, thickness)?;
    let alpha = a / radius;
    let minimum_length_ratio = 2.0 * (radius / thickness).sqrt();
    if !positive(length)
        || radius / thickness < 10.0
        || length / radius < minimum_length_ratio
        || !(poisson > -1.0 && poisson < 0.5)
        || alpha >= 3.0 * PI / 4.0
    {
        return None;
    }
    let k = (thickness / radius).sqrt() / (12.0 * (1.0 - poisson.powi(2))).sqrt().sqrt();
    let b = alpha / (2.0 * k);
    let beta = if b <= 1.0 {
        1.0 + (PI / 16.0 - 0.0293 * b) * b.powi(2)
    } else {
        // Sanders' interpolation uses b outside the inner radical.
        (8.0_f64.sqrt() * b / PI).sqrt() + (0.179 / b).powf(0.885)
    };
    let delta = if alpha == 0.0 {
        0.0
    } else {
        let cot = alpha.tan().recip();
        let root_two = 2.0_f64.sqrt();
        let h_over_alpha =
            root_two / (alpha / ((PI - alpha) / root_two).tan() + root_two * (alpha * cot));
        h_over_alpha * (1.0 - alpha * cot) / 2.0
    };
    // Substitute b = alpha/(2k) into I0/(2π alpha) before evaluation.
    let f_squared = beta.powi(2) + (8.0_f64.sqrt() / PI) * b * delta * (2.0 + delta);
    let factor = PI.sqrt() * a.sqrt() * f_squared.sqrt();
    positive(factor).then_some(factor)
}

// USAF Damage Tolerant Design Handbook, Table 11.3.13, solid round bar in tension.
fn round_bar_tension_factor(depth: f64, diameter: f64) -> f64 {
    let beta = PI / 2.0 * (depth / diameter);
    let tan_ratio = if beta == 0.0 { 1.0 } else { beta.tan() / beta };
    let correction = (0.92 * 2.0 / PI) * tan_ratio.sqrt() / beta.cos();
    let shape = 0.752 + 1.286 * beta + 0.37 * (1.0 - beta.sin()).powi(3);
    correction * shape * PI.sqrt() * depth.sqrt()
}

// Tada's uniform tension solution, NASA NTRS 20240000858 equation (1).
fn edge_factor(a: f64, width: f64) -> f64 {
    let fraction = a / width;
    let angle = PI / 2.0 * fraction;
    let tan_ratio = if angle == 0.0 {
        1.0
    } else {
        angle.tan() / angle
    };
    a.sqrt()
        * (PI * tan_ratio).sqrt()
        * (0.752 + 2.02 * fraction + 0.37 * (1.0 - angle.sin()).powi(3))
        / angle.cos()
}

// Appendix B, SC01, with the area and inertia divided by Wt and Wt³.
fn surface_net_section_factor(depth: f64, half_span: f64) -> f64 {
    let removed = PI / 4.0 * depth * half_span;
    let crack_centroid = 4.0 * depth / (3.0 * PI);
    let centroid = (0.5 - removed * crack_centroid) / (1.0 - removed);
    let offset = centroid - crack_centroid;
    let ellipse_inertia = (0.25 - 16.0 / (9.0 * PI.powi(2))) * depth.powi(2);
    let inertia =
        1.0 / 12.0 + (centroid - 0.5).powi(2) - removed * (ellipse_inertia + offset.powi(2));
    let fiber = (1.0 - 0.13) * centroid.max(1.0 - centroid);
    1.0 / (1.0 - removed) + removed * offset * fiber / inertia
}

// Newman and Raju, NASA TM-83200, equations 16–25, tension component only.
fn surface_factors(a: f64, c: f64, t: f64, b: f64) -> [f64; 2] {
    let shape = a / c;
    let depth_ratio = a / t;
    let depth_squared = depth_ratio.powi(2);
    let width = (PI * c / (2.0 * b) * depth_ratio.sqrt())
        .cos()
        .sqrt()
        .recip();
    if shape <= 1.0 {
        let q = 1.0 + 1.464 * shape.powf(1.65);
        let m1 = 1.13 - 0.09 * shape;
        let m2 = -0.54 + 0.89 / (0.2 + shape);
        let m3 = 0.5 - 1.0 / (0.65 + shape) + 14.0 * (1.0 - shape).powi(24);
        let common =
            (PI * a / q).sqrt() * (m1 + m2 * depth_squared + m3 * depth_ratio.powi(4)) * width;
        return [common, common * (1.1 + 0.35 * depth_squared) * shape.sqrt()];
    }
    let inverse_shape = c / a;
    let q = 1.0 + 1.464 * inverse_shape.powf(1.65);
    let m1 = inverse_shape.sqrt() * (1.0 + 0.04 * inverse_shape);
    let m2 = 0.2 * inverse_shape.powi(4);
    let m3 = -0.11 * inverse_shape.powi(4);
    let common = (PI * a / q).sqrt() * (m1 + m2 * depth_squared + m3 * depth_ratio.powi(4)) * width;
    [
        common * inverse_shape.sqrt(),
        common * (1.1 + 0.35 * inverse_shape * depth_squared),
    ]
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
        self.rate_for_length(Some(length), k_min, k_max, 1.0)
    }

    pub(crate) fn rate_for_length(
        self,
        length: Option<f64>,
        k_min: f64,
        k_max: f64,
        range_factor: f64,
    ) -> Result<f64, GrowthError> {
        if !self.valid() {
            return Err(GrowthError::Material);
        }
        if !k_min.is_finite()
            || k_min < 0.0
            || !positive(k_max)
            || k_min >= k_max
            || k_max >= self.toughness
        {
            return Err(GrowthError::Load);
        }
        // NASGRO 3.0 uses 0.0015 in (0.0381 mm) as the intrinsic crack length.
        let length_factor = match length {
            Some(length) if positive(length) => (length / (length + 0.0381)).sqrt(),
            Some(_) => return Err(GrowthError::Load),
            None => 1.0,
        };
        let delta = range_factor * (k_max - k_min);
        let ratio = k_min / k_max;
        let (a0, opening) = self.opening(ratio);
        let closure = (1.0 - opening) / (1.0 - ratio);
        let threshold = self.threshold_zero * length_factor
            / (closure / (1.0 - a0)).powf(1.0 + self.threshold_fanning * ratio);
        if self.p > 0.0 && delta <= threshold {
            return Ok(0.0);
        }
        let threshold_term = if self.p == 0.0 {
            1.0
        } else {
            (1.0 - threshold / delta).powf(self.p)
        };
        let effective_delta = range_factor * (1.0 - opening) * k_max;
        let rate = self.coefficient * effective_delta.powf(self.exponent) * threshold_term
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
            // NASFLA Eq. (2.10) corrects the growth range at the SC01 c-tip.
            let range_factor = if index == 1 && matches!(plate, Plate::Surface { .. }) {
                let r_squared = (load.minimum / load.maximum).powi(2);
                0.9 + r_squared * (0.2 - 0.1 * r_squared)
            } else {
                1.0
            };
            rate[index] = material.rate_for_length(
                Some(sizes[index]),
                k_min[index],
                k_max[index],
                range_factor,
            )?;
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
            .ok_or(GrowthError::HistoryCapacity)?;
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
                return Err(GrowthError::CycleResolution {
                    block: index + 1,
                    cycles: self.elapsed,
                });
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
            Err(GrowthError::TrialBudget {
                block: index + 1,
                cycles: self.elapsed,
                limit: self.control.max_trials,
            })
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
