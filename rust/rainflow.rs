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

#[cfg(test)]
mod tests {
    use super::{CrossingError, find_crossings};

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
