//! Shared functionality between integration tests.

use std::io::Cursor;

use csv::ReaderBuilder;
use ndarray::Array2;
use ndarray_csv::Array2Reader as _;

/// Load the signals from a text file.
#[must_use]
pub fn load_signals_fixture(file: &'static str) -> Array2<f64> {
    // Read CSV
    let mut cursor = Cursor::new(file);
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(&mut cursor);

    // Convert to array
    reader
        .deserialize_array2_dynamic()
        .expect("Error deserializing CSV into array")
}
