//! Benchmark different configurations.

use std::io::Cursor;

use criterion::{BenchmarkId, Criterion};
use csv::ReaderBuilder;
use ndarray::Array2;
use ndarray_csv::Array2Reader as _;
use pelt::{Pelt, SegmentCostFunction};

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

            // Benchmark
            group.bench_with_input(
                BenchmarkId::from_parameter(parameter),
                &signal.view(),
                |benchmark, signal| {
                    benchmark.iter(|| {
                        // Run the benchmark
                        let result = Pelt::new()
                            .with_segment_cost_function(std::hint::black_box(segment_cost_function))
                            .predict(std::hint::black_box(signal), std::hint::black_box(10.0));
                        let _ = std::hint::black_box(result);
                    });
                },
            );
        }

        group.finish();
    }
}

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
