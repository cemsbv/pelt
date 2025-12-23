import numpy as np
from scipy import stats
from pelt import predict
from ruptures.detection import Pelt
from ruptures.datasets import pw_normal

# Generate random signal data with 10 groups
def benchmark(segment, data, changepoints) :
    # Generate random data
    (signal, _) = pw_normal(data, changepoints)

    def pelt():
        predict(signal, penalty=10, segment_cost_function=segment, sum_method="naive")

    def ruptures():
        Pelt(model=segment).fit_predict(signal, pen=10)

    return (ruptures, pelt, f"{segment.upper()} | {data} | {changepoints}")

__benchmarks__ = [
    benchmark("l1", 100, 2),
    benchmark("l1", 100, 10),
    benchmark("l1", 1000, 2),
    benchmark("l1", 1000, 10),
    benchmark("l1", 1000, 100),

    benchmark("l2", 100, 2),
    benchmark("l2", 100, 10),
    benchmark("l2", 1000, 2),
    benchmark("l2", 1000, 10),
    benchmark("l2", 1000, 100),
]

