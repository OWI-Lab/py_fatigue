#[derive(Debug, PartialEq, Eq)]
pub enum CrossingError {
    OutputTooSmall,
    IndexOutOfRange,
}

pub fn find_crossings(values: &[i8], output: &mut [i64]) -> Result<usize, CrossingError> {
    if output.len() < values.len() {
        return Err(CrossingError::OutputTooSmall);
    }
    let Some(first) = values.first() else {
        return Ok(0);
    };
    let mut direction = first.signum();
    let mut count = 0;
    let mut start = 0;

    if direction == 0 {
        for (index, &value) in values.iter().enumerate().skip(1) {
            if value != 0 {
                output[0] = i64::try_from(index - 1).map_err(|_| CrossingError::IndexOutOfRange)?;
                direction = value.signum();
                start = index;
                count = 1;
                break;
            }
        }
    }

    for (index, pair) in values.windows(2).enumerate().skip(start) {
        let up = direction == -1 && pair[0] <= 0 && pair[1] > 0;
        let down = direction == 1 && pair[0] >= 0 && pair[1] < 0;
        if up || down {
            let slot = output.get_mut(count).ok_or(CrossingError::OutputTooSmall)?;
            *slot = i64::try_from(index).map_err(|_| CrossingError::IndexOutOfRange)?;
            direction = -direction;
            count += 1;
        }
    }
    Ok(count)
}

#[derive(Debug, PartialEq, Eq)]
pub enum RainflowError {
    Workspace,
    Times,
}

type Cycle = [f64; 5];

fn cycle_between(
    values: &[f64],
    times: Option<&[f64]>,
    first: usize,
    second: usize,
    count: f64,
) -> Option<Cycle> {
    let amplitude = (values[second] - values[first]).abs() / 2.0;
    (amplitude > 0.0).then(|| {
        let mean = (values[first] + values[second]) / 2.0;
        let [start, period] = times.map_or([0.0; 2], |t| [t[first], 2.0 * (t[second] - t[first])]);
        [amplitude, mean, count, start, period]
    })
}

/// Count ASTM three-point cycles in order, followed by residual half cycles.
pub fn count_astm(
    values: &[f64],
    times: Option<&[f64]>,
    pending: &mut [usize],
    output: &mut [Cycle],
) -> Result<usize, RainflowError> {
    if pending.len() < values.len() || output.len() < values.len() {
        return Err(RainflowError::Workspace);
    }
    if times.is_some_and(|t| t.len() < values.len()) {
        return Err(RainflowError::Times);
    }
    let mut queued = 0;
    let mut written = 0;
    let mut full_cycles = 0;
    for incoming in 0..values.len() {
        pending[queued] = incoming;
        queued += 1;
        // A closure removes at least one pending reversal.
        let reversals_to_check = queued;
        for _ in 2..reversals_to_check {
            let [first, second, last] = pending[queued - 3..queued] else {
                return Err(RainflowError::Workspace);
            };
            let prior = (values[second] - values[first]).abs();
            let latest = (values[last] - values[second]).abs();
            if prior.partial_cmp(&latest).is_none_or(|order| order.is_gt()) {
                break;
            }
            let is_full = queued > 3;
            if let Some(row) = cycle_between(
                values,
                times,
                first,
                second,
                if is_full { 1.0 } else { 0.5 },
            ) {
                output[written] = row;
                written += 1;
                full_cycles += usize::from(is_full);
            }
            if is_full {
                pending[queued - 3] = last;
                queued -= 2;
            } else {
                pending[..2].copy_from_slice(&[second, last]);
                queued = 2;
            }
            if queued < 3 {
                break;
            }
        }
    }
    for pair in pending[..queued].windows(2) {
        if let Some(row) = cycle_between(values, times, pair[0], pair[1], 0.5) {
            output[written] = row;
            written += 1;
        }
    }
    // Py-Fatigue 2.1.1 includes zero rows when duplicate reversals
    // reduce the emitted count without reducing the legacy output length.
    let count = values.len().saturating_sub(1 + full_cycles);
    output
        .get_mut(written..count)
        .ok_or(RainflowError::Workspace)?
        .fill([0.0; 5]);
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::{CrossingError, RainflowError, count_astm, find_crossings};

    #[test]
    fn astm_keeps_cycle_order_timestamps_and_duplicate_rows() {
        let reversals = [0.0, 5.0, 1.0, 1.0, 6.0];
        let times = [2.0, 3.0, 5.0, 8.0, 11.0];
        let mut pending = [0; 5];
        let mut cycles = [[f64::NAN; 5]; 5];
        assert_eq!(
            count_astm(&reversals, Some(&times), &mut pending, &mut cycles),
            Ok(4)
        );
        assert_eq!(
            &cycles[..4],
            &[
                [2.5, 2.5, 0.5, 2.0, 2.0],
                [0.5, 5.5, 0.5, 3.0, 16.0],
                [0.0; 5],
                [0.0; 5],
            ]
        );
        assert_eq!(
            count_astm(&reversals, Some(&times[..3]), &mut pending, &mut cycles),
            Err(RainflowError::Times)
        );
        assert_eq!(
            count_astm(&reversals, None, &mut pending[..3], &mut cycles),
            Err(RainflowError::Workspace)
        );
    }

    #[test]
    fn preserves_plateau_indices() {
        let mut output = [0; 6];
        assert_eq!(find_crossings(&[0, 0, -1, 0, 1, -1], &mut output), Ok(3));
        assert_eq!(&output[..3], &[1, 3, 4]);
    }

    #[test]
    fn handles_empty_constant_and_short_buffers() {
        let mut output = [0; 2];
        assert_eq!(find_crossings(&[], &mut output), Ok(0));
        assert_eq!(find_crossings(&[0, 0], &mut output), Ok(0));
        assert_eq!(
            find_crossings(&[1, -1, 1], &mut output),
            Err(CrossingError::OutputTooSmall)
        );
    }
}
