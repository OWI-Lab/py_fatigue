use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyAssertionError, PyMemoryError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;

use crate::crack_growth::{self, Controls, GrowthError, Load, Nasgro, Plate};
use crate::curves::{self, Curve, CurveKind};
use crate::legacy_growth::{Geometry, LegacyGrowth, Stop};
use crate::rainflow;
use crate::stress_life::{self, Goodman, StressLifeError};

#[pyfunction]
fn find_crossings<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, i8>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let values = values.as_slice()?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| PyMemoryError::new_err("could not allocate crossing indices"))?;
    output.resize(values.len(), 0);
    let count = rainflow::find_crossings(values, &mut output)
        .map_err(|error| PyValueError::new_err(format!("invalid crossing indices: {error:?}")))?;
    output.truncate(count);
    Ok(output.into_pyarray(py))
}

#[pyfunction]
fn evaluate_curve<'py>(
    py: Python<'py>,
    kind: u8,
    values: PyReadonlyArray1<'_, f64>,
    slopes: PyReadonlyArray1<'_, f64>,
    intercepts: PyReadonlyArray1<'_, f64>,
    knees: PyReadonlyArray1<'_, f64>,
    limits: [f64; 2],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kind = match kind {
        0 => CurveKind::StressToCycles,
        1 => CurveKind::CyclesToStress,
        2 => CurveKind::SifToGrowth,
        3 => CurveKind::GrowthToSif,
        _ => return Err(PyValueError::new_err("unknown curve operation")),
    };
    let curve = Curve::new(
        slopes.as_slice()?,
        intercepts.as_slice()?,
        knees.as_slice()?,
        limits,
        kind,
    )
    .map_err(curve_error)?;
    let values = values.as_slice()?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| PyMemoryError::new_err("could not allocate curve outputs"))?;
    output.resize(values.len(), 0.0);
    curve.fill(values, &mut output).map_err(curve_error)?;
    Ok(output.into_pyarray(py))
}

fn curve_error(error: curves::CurveError) -> PyErr {
    PyAssertionError::new_err(format!("invalid curve inputs: {error:?}"))
}

#[pyfunction]
fn sn_intersection(
    slopes: PyReadonlyArray1<'_, f64>,
    intercepts: PyReadonlyArray1<'_, f64>,
    knees: PyReadonlyArray1<'_, f64>,
    limits: [f64; 2],
    stress_weight: [f64; 2],
    bounds: [f64; 2],
    control: (f64, usize),
) -> PyResult<f64> {
    let curve = Curve::new(
        slopes.as_slice()?,
        intercepts.as_slice()?,
        knees.as_slice()?,
        limits,
        CurveKind::CyclesToStress,
    )
    .map_err(curve_error)?;
    curves::sn_intersection(
        &curve,
        stress_weight[0],
        stress_weight[1],
        bounds[0],
        bounds[1],
        control.0,
        control.1,
    )
    .map_err(curve_error)
}

#[pyfunction]
fn hollow_cylinder_factor(depth: f64, dimensions: [f64; 3]) -> f64 {
    cylinder(dimensions).factor(depth)
}

fn cylinder(dimensions: [f64; 3]) -> Geometry {
    let [diameter, thickness, width_to_depth] = dimensions;
    Geometry::HollowCylinder {
        diameter,
        thickness,
        width_to_depth,
    }
}

type GrowthOutput<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    &'static str,
);

#[pyfunction]
fn legacy_growth<'py>(
    py: Python<'py>,
    loads: (PyReadonlyArray1<'_, f64>, PyReadonlyArray1<'_, f64>),
    curve: (
        PyReadonlyArray1<'_, f64>,
        PyReadonlyArray1<'_, f64>,
        PyReadonlyArray1<'_, f64>,
        [f64; 2],
    ),
    initial_depth: f64,
    dimensions: Option<[f64; 3]>,
) -> PyResult<GrowthOutput<'py>> {
    let input = LegacyGrowth {
        stress: loads.0.as_slice()?,
        counts: loads.1.as_slice()?,
        slopes: curve.0.as_slice()?,
        intercepts: curve.1.as_slice()?,
        knees: curve.2.as_slice()?,
        limits: curve.3,
        geometry: dimensions.map_or(Geometry::Infinite, cylinder),
        initial_depth,
    };
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.stress.len())
        .map_err(|_| PyMemoryError::new_err("could not allocate crack history"))?;
    output.resize(input.stress.len(), [0.0; 3]);
    let (count, reason) = input.simulate(&mut output).map_err(curve_error)?;
    let extract = |index| {
        let values: Vec<_> = output[..count].iter().map(|row| row[index]).collect();
        values.into_pyarray(py)
    };
    let reason = match reason {
        Stop::Spectrum => "Fatigue spectrum applied w/o failure. Stopping calculation",
        Stop::Thickness => {
            "Crack depth is greater than or equal to the thickness.Stopping calculation."
        }
        Stop::Critical => "Critical SIF reached. Stopping calculation.",
        Stop::Increment => {
            "Crack size increased by more than 10x in the last iteration. Stopping calculation."
        }
    };
    Ok((extract(0), extract(1), extract(2), reason))
}

#[pyfunction]
fn solve_goodman(
    initial: f64,
    amplitude: f64,
    ratio_in: f64,
    ratio_out: f64,
    ultimate: f64,
    exponent: f64,
) -> PyResult<f64> {
    Goodman {
        amplitude,
        ratio_in,
        ratio_out,
        ultimate,
        exponent,
    }
    .solve(initial)
    .map_err(|error| match error {
        StressLifeError::StaticLoad => {
            PyAssertionError::new_err("Input load ratio equals 1, meaning static load.")
        }
        StressLifeError::ZeroDerivative => {
            PyZeroDivisionError::new_err("Goodman solver derivative is zero")
        }
        StressLifeError::Shape => PyValueError::new_err("invalid Goodman input dimensions"),
    })
}

type TheilOutput<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    f64,
);

#[pyfunction]
fn theil_history<'py>(
    py: Python<'py>,
    stress: PyReadonlyArray1<'_, f64>,
    counts: PyReadonlyArray1<'_, f64>,
    lives: PyReadonlyArray1<'_, f64>,
    sizes: Vec<usize>,
) -> PyResult<TheilOutput<'py>> {
    let total = sizes
        .iter()
        .try_fold(0_usize, |acc, &size| acc.checked_add(size))
        .ok_or_else(|| PyValueError::new_err("Theil history size overflow"))?;
    let mut cycles = Vec::new();
    let mut history = Vec::new();
    for buffer in [&mut cycles, &mut history] {
        buffer
            .try_reserve_exact(total)
            .map_err(|_| PyMemoryError::new_err("could not allocate Theil history"))?;
        buffer.resize(total, 0.0);
    }
    let (effective, final_cycles) = stress_life::theil_history(
        stress.as_slice()?,
        counts.as_slice()?,
        lives.as_slice()?,
        &sizes,
        &mut cycles,
        &mut history,
    )
    .map_err(|_| PyValueError::new_err("inconsistent Theil input lengths"))?;
    Ok((
        cycles.into_pyarray(py),
        history.into_pyarray(py),
        effective,
        final_cycles,
    ))
}

fn nasgro(parameters: [f64; 9]) -> Nasgro {
    let [
        coefficient,
        exponent,
        p,
        q,
        threshold_zero,
        threshold_fanning,
        toughness,
        constraint,
        peak_to_flow,
    ] = parameters;
    Nasgro {
        coefficient,
        exponent,
        p,
        q,
        threshold_zero,
        threshold_fanning,
        toughness,
        constraint,
        peak_to_flow,
    }
}

fn plate(half_width: f64, thickness: Option<f64>) -> Plate {
    match thickness {
        Some(thickness) => Plate::Surface {
            half_width,
            thickness,
        },
        None => Plate::Center { half_width },
    }
}

fn growth_error(error: GrowthError) -> PyErr {
    match error {
        GrowthError::Budget => pyo3::exceptions::PyRuntimeError::new_err(
            "crack growth exhausted the step budget or cycle resolution",
        ),
        GrowthError::NonFinite => pyo3::exceptions::PyArithmeticError::new_err(
            "crack growth produced a non-finite numerical result",
        ),
        _ => PyValueError::new_err(format!("invalid crack growth input: {error:?}")),
    }
}

#[pyfunction]
fn plate_factors(half_width: f64, thickness: Option<f64>, initial: [f64; 2]) -> PyResult<[f64; 2]> {
    plate(half_width, thickness)
        .factors(initial)
        .ok_or_else(|| growth_error(GrowthError::Geometry))
}

#[pyfunction]
fn nasgro_rate(parameters: [f64; 9], length: f64, k_min: f64, k_max: f64) -> PyResult<f64> {
    nasgro(parameters)
        .growth_rate(length, k_min, k_max)
        .map_err(growth_error)
}

type PointData = (f64, [f64; 2], [f64; 2], [f64; 2], usize);

#[pyfunction]
fn ordered_growth(
    section: (f64, Option<f64>),
    initial: [f64; 2],
    parameters: [f64; 9],
    schedule: Vec<[f64; 3]>,
    controls: (f64, f64, f64, usize),
) -> PyResult<(Vec<PointData>, &'static str)> {
    let loads: Vec<_> = schedule
        .into_iter()
        .map(|[minimum, maximum, cycles]| Load {
            minimum,
            maximum,
            cycles,
        })
        .collect();
    let (max_step, length_tolerance, cycle_tolerance, max_trials) = controls;
    let controls = Controls {
        max_step,
        length_tolerance,
        cycle_tolerance,
        max_trials,
    };
    let capacity = max_trials
        .checked_add(loads.len())
        .ok_or_else(|| growth_error(GrowthError::Budget))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| PyMemoryError::new_err("could not allocate crack history"))?;
    output.resize(capacity, crack_growth::Point::default());
    let (written, stop) = crack_growth::simulate(
        plate(section.0, section.1),
        initial,
        nasgro(parameters),
        &loads,
        &controls,
        &mut output,
    )
    .map_err(growth_error)?;
    let stop = match stop {
        crack_growth::Stop::Spectrum => "spectrum_exhausted",
        crack_growth::Stop::Toughness => "toughness",
        crack_growth::Stop::GeometryLimit => "geometry_limit",
    };
    let history = output[..written]
        .iter()
        .map(|point| {
            (
                point.cycles,
                point.sizes,
                point.k_min,
                point.k_max,
                point.block,
            )
        })
        .collect();
    Ok((history, stop))
}

#[pymodule(gil_used = true)]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(find_crossings, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_curve, module)?)?;
    module.add_function(wrap_pyfunction!(sn_intersection, module)?)?;
    module.add_function(wrap_pyfunction!(hollow_cylinder_factor, module)?)?;
    module.add_function(wrap_pyfunction!(legacy_growth, module)?)?;
    module.add_function(wrap_pyfunction!(solve_goodman, module)?)?;
    module.add_function(wrap_pyfunction!(theil_history, module)?)?;
    module.add_function(wrap_pyfunction!(plate_factors, module)?)?;
    module.add_function(wrap_pyfunction!(nasgro_rate, module)?)?;
    module.add_function(wrap_pyfunction!(ordered_growth, module)?)
}
