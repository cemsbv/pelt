//! Example of reading a 1D txt file.

use std::{error::Error, fs::File};

use csv::ReaderBuilder;
use ndarray_csv::Array2Reader as _;
use pelt::{Pelt, SegmentCostFunction};

pub fn main() -> Result<(), Box<dyn Error>> {
    // Try to read each argument as a file
    for arg in std::env::args().skip(1) {
        eprintln!("Reading file '{arg}'");

        // Read CSV file
        let mut file = File::open(arg)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .from_reader(&mut file);

        // Convert to array
        let signal = reader.deserialize_array2_dynamic()?;

        // Run the algorithm
        eprintln!("L1:");
        match Pelt::new()
            .with_segment_cost_function(SegmentCostFunction::L1)
            .predict(&signal, 10.0_f64)
        {
            Ok(result) => println!("{result:?}"),
            // Print the error
            Err(err) => eprintln!("Error running PELT: {err}"),
        }

        eprintln!("L2:");
        match Pelt::new()
            .with_segment_cost_function(SegmentCostFunction::L2)
            .predict(&signal, 10.0_f64)
        {
            Ok(result) => println!("{result:?}"),
            // Print the error
            Err(err) => eprintln!("Error running PELT: {err}"),
        }
    }

    Ok(())
}
