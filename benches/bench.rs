//! Benchmark different configurations.

use std::io::Cursor;

use criterion::{BenchmarkId, Criterion};
use csv::ReaderBuilder;
// use divan::Bencher;
use ndarray::Array2;
use ndarray_csv::Array2Reader as _;
use pelt::{Kahan, Naive, Pelt, SegmentCostFunction};

/// Benchmark the different groups and cases.
fn benchmark(criterion: &mut Criterion) {
    // Create different benchmark groups based on the files to test
    for (name, signal_data) in [
        ("small", include_str!("../tests/signals-small.csv")),
        ("large", include_str!("../tests/signals-large.csv")),
        ("2D", include_str!("../tests/normal-10.csv")),
    ] {
        let mut group = criterion.benchmark_group(name);

        let signal = load_signals_fixture(signal_data);

        // Benchmark each segment cost function
        for segment_cost_function in [SegmentCostFunction::L1, SegmentCostFunction::L2] {
            let parameter = match segment_cost_function {
                SegmentCostFunction::L1 => "L1",
                SegmentCostFunction::L2 => "L2",
            };

            // Benchmark "naive"
            group.bench_with_input(
                BenchmarkId::new("Naive", parameter),
                &signal.view(),
                |benchmark, signal| {
                    benchmark.iter(|| {
                        // Run the benchmark
                        let result = Pelt::new()
                            .with_segment_cost_function(std::hint::black_box(segment_cost_function))
                            .predict::<Naive>(
                                std::hint::black_box(signal),
                                std::hint::black_box(10.0),
                            );
                        let _ = std::hint::black_box(result);
                    });
                },
            );

            // Benchmark "kahan"
            group.bench_with_input(
                BenchmarkId::new("Kahan", parameter),
                &signal.view(),
                |benchmark, signal| {
                    benchmark.iter(|| {
                        // Run the benchmark
                        let result = Pelt::new()
                            .with_segment_cost_function(std::hint::black_box(segment_cost_function))
                            .predict::<Kahan>(
                                std::hint::black_box(signal),
                                std::hint::black_box(10.0),
                            );
                        let _ = std::hint::black_box(result);
                    });
                },
            );
        }

        group.finish();
    }
}

/*
/// Benchmark the small signals file.
#[divan::bench(args = [SegmentCostFunction::L1, SegmentCostFunction::L2], types = [Kahan, Naive])]
fn small<S: Sum<f64> + Send + Sync>(bencher: Bencher, segment_cost_function: SegmentCostFunction) {
    bencher
        .with_inputs(|| load_signals_fixture(include_str!("../tests/signals-small.csv")))
        .bench_local_values(move |array: Array2<f64>| {
            let result = Pelt::new()
                .with_segment_cost_function(segment_cost_function)
                .predict::<S>(divan::black_box(array.view()), 10.0);
            divan::black_box_drop(result);
        });
}

/// Benchmark the large signals file.
#[divan::bench(args = [SegmentCostFunction::L1, SegmentCostFunction::L2], types = [Kahan, Naive])]
fn large<S: Sum<f64> + Send + Sync>(bencher: Bencher, segment_cost_function: SegmentCostFunction) {
    bencher
        .with_inputs(|| load_signals_fixture(include_str!("../tests/signals-large.csv")))
        .bench_local_values(move |array: Array2<f64>| {
            let result = Pelt::new()
                .with_segment_cost_function(segment_cost_function)
                .predict::<S>(divan::black_box(array.view()), 10.0);
            divan::black_box_drop(result);
        });
}

/// Benchmark the large signals file.
#[divan::bench(args = [SegmentCostFunction::L1, SegmentCostFunction::L2], types = [Kahan, Naive])]
fn small_2d<S: Sum<f64> + Send + Sync>(
    bencher: Bencher,
    segment_cost_function: SegmentCostFunction,
) {
    bencher
        .with_inputs(|| load_signals_fixture(include_str!("../tests/normal-10.csv")))
        .bench_local_values(move |array: Array2<f64>| {
            let result = Pelt::new()
                .with_segment_cost_function(segment_cost_function)
                .predict::<S>(divan::black_box(array.view()), 3.0);
            divan::black_box_drop(result);
        });
}
*/

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

criterion::criterion_group!(benches, benchmark);
criterion::criterion_main!(benches);
