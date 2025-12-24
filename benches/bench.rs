//! Benchmark different configurations.

use std::io::Cursor;

use criterion::{BenchmarkId, Criterion, Throughput};
use csv::ReaderBuilder;
use fearless_simd::Level;
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
        let signal = load_signals_fixture(signal_data);

        let mut group = criterion.benchmark_group(name);

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

    {
        // Benchmark cost function
        let mut group = criterion.benchmark_group("cost");

        let signal = load_signals_fixture(include_str!("../tests/normal-10.csv"));

        // Benchmark each segment cost function
        for segment_cost_function in [SegmentCostFunction::L1, SegmentCostFunction::L2] {
            let parameter = match segment_cost_function {
                SegmentCostFunction::L1 => "L1",
                SegmentCostFunction::L2 => "L2",
            };

            // Benchmark these ranges
            for size in [1_usize, 4, 10, 32, 100] {
                group.throughput(Throughput::Elements(size as u64));

                group.bench_with_input(
                    BenchmarkId::new(parameter, size),
                    &(Level::new(), signal.view()),
                    |benchmark, (simd_level, signal)| {
                        benchmark.iter(|| {
                            // Run the benchmark
                            segment_cost_function.loss(
                                std::hint::black_box(*simd_level),
                                std::hint::black_box(signal),
                                std::hint::black_box(0..size),
                            )
                        });
                    },
                );
            }
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
