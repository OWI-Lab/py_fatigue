#![forbid(unsafe_code)]

pub mod crack_growth;
pub mod curves;
pub mod legacy_growth;
pub mod rainflow;
pub mod stress_life;

// The numerical core also builds and runs without a Python runtime.
#[cfg(feature = "python")]
mod python;
