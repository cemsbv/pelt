from pelt import predict
import timeit
from statistics import fmean
from ruptures.detection import Pelt
from ruptures.datasets import pw_wavy, pw_normal

def format_time(time):
    if time > 1:
        return f"{time:.3f} s"
    elif time < 0.001:
        time *= 1000000
        return f"{time:.3f} μs"
    else:
        time *= 1000
        return f"{time:.3f} ms"
    

# Generate random signal data with 10 groups
def benchmark(dim, segment, data, repeat_ruptures, repeat_pelt):
    # Generate random data
    if dim == 1:
        (signal, _) = pw_wavy(n_samples=data, n_bkps=10, seed=0)
    else:
        (signal, _) = pw_normal(n_samples=data, n_bkps=10, seed=0)

    def pelt():
        predict(signal, penalty=10, segment_cost_function=segment)

    def ruptures():
        Pelt(model=segment).fit_predict(signal, pen=10)

    # Test ruptures N times
    ruptures_result = timeit.repeat(ruptures, repeat=repeat_ruptures, number=1)
    # Test pelt N * 100 times
    pelt_result = timeit.repeat(pelt, repeat=repeat_pelt, number=1)

    # Take the mean as millisecond
    ruptures_mean = fmean(ruptures_result)
    pelt_mean = fmean(pelt_result)

    # Calculate the difference
    delta_mean = ruptures_mean / pelt_mean

    # Print table row
    bold_treshold = 1000
    print(f"| _{segment.upper()}_ | _{data}_ | _{dim}D_ | {format_time(pelt_mean)} | {format_time(ruptures_mean)} | {"**" if delta_mean > bold_treshold else ""}{delta_mean:.1f}x{"**" if delta_mean > bold_treshold else ""} |")
    

def main():
    print("| Cost Function | Data Points | Data Dimension | Mean `pelt` | Mean `ruptures` | Times Faster |")
    print("| -- | -- | -- | -- | -- | -- |")

    benchmark(1, "l2", 100, 1000, 100000)
    benchmark(2, "l2", 100, 1000, 100000)
    benchmark(1, "l2", 1000, 10, 10000)
    benchmark(2, "l2", 1000, 10, 10000)
    benchmark(1, "l2", 10000, 2, 100)
    benchmark(2, "l2", 10000, 2, 100)

    benchmark(1, "l1", 100, 1000, 10000)
    benchmark(2, "l1", 100, 1000, 10000)
    benchmark(1, "l1", 1000, 10, 1000)
    benchmark(2, "l1", 1000, 10, 1000)
    benchmark(1, "l1", 10000, 2, 10)
    benchmark(2, "l1", 10000, 2, 10)

if __name__ == "__main__":
    main()
