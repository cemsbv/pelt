//! Example of reading a 1D txt file.

use std::error::Error;

use ndarray::Array2;
use pelt::{Naive, Pelt, SegmentCostFunction};

pub fn main() -> Result<(), Box<dyn Error>> {
    // Try to read each argument as a file
    for arg in std::env::args().skip(1) {
        eprintln!("Reading file '{arg}'");

        let file = std::fs::read_to_string(arg)?;

        // Load the signal dataset
        let data = file
            .lines()
            .enumerate()
            .filter_map(|(line, float)| {
                float
                    .parse::<f64>()
                    .inspect_err(|err| {
                        eprintln!(
                            "Test value '{float}' on line {} is not a valid float: {err}",
                            line + 1
                        )
                    })
                    .ok()
            })
            .collect::<Vec<_>>();

        // Convert to ndarray
        let mut signal = Array2::zeros((data.len(), 1));
        signal
            .iter_mut()
            .zip(data)
            .for_each(|(item, data)| *item = data);

        // Run the algorithm
        eprintln!("L1:");
        match Pelt::new()
            .with_segment_cost_function(SegmentCostFunction::L1)
            .predict::<Naive<_>, _>(&signal, 20.0_f64)
        {
            Ok(result) => println!("{result:?}"),
            // Print the error
            Err(err) => eprintln!("Error running PELT: {err}"),
        }

        eprintln!("L2:");
        match Pelt::new()
            .with_segment_cost_function(SegmentCostFunction::L2)
            .predict::<Naive<_>, _>(&signal, 20.0_f64)
        {
            Ok(result) => println!("{result:?}"),
            // Print the error
            Err(err) => eprintln!("Error running PELT: {err}"),
        }
    }

    Ok(())
}
