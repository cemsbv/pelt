//! Benchmark different configurations.

use divan::Bencher;
use ndarray::Array2;
use pelt::{Kahan, Naive, Pelt, SegmentCostFunction, Sum};

/// Benchmark the small signals file.
#[divan::bench(args = [SegmentCostFunction::L1, SegmentCostFunction::L2], types = [Kahan<f64>, Naive<f64>])]
fn small<S: Sum<f64>>(bencher: Bencher, segment_cost_function: SegmentCostFunction) {
    bencher
        .with_inputs(|| load_signals_fixture(include_str!("../tests/signals.txt")))
        .bench_local_values(move |array: Array2<f64>| {
            let result = Pelt::new()
                .with_segment_cost_function(segment_cost_function)
                .predict::<S, _>(divan::black_box(array.view()), 10.0);
            divan::black_box_drop(result);
        });
}

/// Benchmark the large signals file.
#[divan::bench(args = [SegmentCostFunction::L1, SegmentCostFunction::L2], types = [Kahan<f64>, Naive<f64>])]
fn large<S: Sum<f64>>(bencher: Bencher, segment_cost_function: SegmentCostFunction) {
    bencher
        .with_inputs(|| load_signals_fixture(include_str!("../tests/signals-large.txt")))
        .bench_local_values(move |array: Array2<f64>| {
            let result = Pelt::new()
                .with_segment_cost_function(segment_cost_function)
                .predict::<S, _>(divan::black_box(array.view()), 10.0);
            divan::black_box_drop(result);
        });
}

/// Load the signals from a text file.
fn load_signals_fixture(file: &'static str) -> Array2<f64> {
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

fn main() {
    divan::main();
}
