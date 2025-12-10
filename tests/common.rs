//! Shared functionality between integration tests.

use ndarray::Array2;

/// Load the signals from a text file.
#[must_use]
pub fn load_signals_fixture(file: &'static str) -> Array2<f64> {
    // Load the signal dataset
    let data = file
        .lines()
        .enumerate()
        .map(|(line, float)| {
            float.parse::<f64>().unwrap_or_else(|_| {
                panic!(
                    "Test value '{float}' on line {} is not a valid float",
                    line + 1
                )
            })
        })
        .collect::<Vec<_>>();

    // Convert to ndarray
    let mut array = Array2::zeros((data.len(), 1));
    array
        .iter_mut()
        .zip(data)
        .for_each(|(item, data)| *item = data);

    array
}
