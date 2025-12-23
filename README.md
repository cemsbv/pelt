[![Crates.io][ci]][cl] [![pypi][pi]][pl] ![MPL-2.0][li] [![docs.rs][di]][dl] ![ci][bci]

[ci]: https://img.shields.io/crates/v/pelt.svg
[cl]: https://crates.io/crates/pelt/
[pi]: https://badge.fury.io/py/pelt.svg
[pl]: https://pypi.org/project/pelt
[li]: https://img.shields.io/crates/l/pelt.svg?maxAge=2592000
[di]: https://docs.rs/pelt/badge.svg
[dl]: https://docs.rs/pelt/
[bci]: https://github.com/cemsbv/pelt/workflows/ci/badge.svg

Changepoint detection with Pruned Exact Linear Time. 

## Usage

### Python

```python
from pelt import predict

predict(signal, penalty=20, segment_cost_function="l1", jump=10, minimum_segment_length=2, sum_method="kahan")
```

### Rust

```rust
use pelt::{Pelt, SegmentCostFunction, Kahan};

// Setup the structure for calculating changepoints
let pelt = Pelt::new()
  .with_jump(NonZero::new(5).expect("Invalid number"))
  .with_minimum_segment_length(NonZero::new(2).expect("Invalid number"))
  .with_segment_cost_function(SegmentCostFunction::L1);

// Do the calculation on a data set
let penalty = 10.0;
// Use more accurate Kahan summation for all math
let result = pelt.predict::<Kahan>(&signal[..], penalty)?;
```

## Run locally

```sh
# Install maturin inside a Python environment
python3 -m venv .env
source .env/bin/activate
pip install maturin numpy

# Create a Python package from the Rust code
maturin develop --features python

# Open an interpreter
python

>>> from pelt import predict
>>> import numpy as np
>>> signal = np.array([np.sin(np.arange(0, 1000, 10))]).transpose()
>>> predict(signal, penalty=20)
```

## Benchmarks

Like all benchmarks, take these with a grain of salt.

### Python

Comparison with [ruptures](https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/):

| Cost Function | Data Points | Change Points | Min | Max | Mean | Min (+) | Max (+) | Mean (+) |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| L1 | 100 | 2 | 0.026 | 0.027 | 0.026 | 0.000 (212.4x) | 0.000 (96.5x) | 0.000 (167.8x) |
| L1 | 100 | 10 | 0.026 | 0.026 | 0.026 | 0.000 (224.0x) | 0.000 (178.7x) | 0.000 (213.0x) |
| L1 | 1000 | 2 | 2.930 | 3.063 | 2.997 | 0.019 (154.2x) | 0.021 (144.1x) | 0.020 (151.9x) |
| L1 | 1000 | 10 | 2.495 | 2.664 | 2.608 | 0.021 (116.6x) | 0.023 (113.8x) | 0.022 (117.6x) |
| L1 | 1000 | 100 | 3.180 | 3.262 | 3.238 | 0.022 (147.5x) | 0.023 (143.4x) | 0.022 (147.4x) |
| L2 | 100 | 2 | 0.015 | 0.016 | 0.016 | 0.000 (555.0x) | 0.000 (450.2x) | 0.000 (528.7x) |
| L2 | 100 | 10 | 0.015 | 0.015 | 0.015 | 0.000 (534.1x) | 0.000 (470.6x) | 0.000 (517.6x) |
| L2 | 1000 | 2 | 0.632 | 0.695 | 0.656 | 0.005 (129.3x) | 0.005 (141.1x) | 0.005 (134.0x) |
| L2 | 1000 | 10 | 1.059 | 1.112 | 1.088 | 0.007 (160.5x) | 0.007 (166.6x) | 0.007 (164.2x) |
| L2 | 1000 | 100 | 0.990 | 1.052 | 1.011 | 0.004 (223.0x) | 0.005 (233.2x) | 0.004 (226.4x) |

<details>

<summary>Command</summary>

```sh
maturin develop --features python --release
richbench benches/
```

</details>

### Rust

```
Timer precision: 20 ns
bench            fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ large                       │               │               │               │         │
│  ├─ Kahan                    │               │               │               │         │
│  │  ├─ L1      41.89 ms      │ 89 ms         │ 45.54 ms      │ 48.45 ms      │ 100     │ 100
│  │  ╰─ L2      8.63 ms       │ 9.518 ms      │ 8.7 ms        │ 8.757 ms      │ 100     │ 100
│  ╰─ Naive                    │               │               │               │         │
│     ├─ L1      43.23 ms      │ 71.51 ms      │ 46.76 ms      │ 48.25 ms      │ 100     │ 100
│     ╰─ L2      8.413 ms      │ 10.71 ms      │ 8.499 ms      │ 8.839 ms      │ 100     │ 100
╰─ small                       │               │               │               │         │
   ├─ Kahan                    │               │               │               │         │
   │  ├─ L1      214.2 µs      │ 234.2 µs      │ 216.4 µs      │ 217.6 µs      │ 100     │ 100
   │  ╰─ L2      21.54 µs      │ 25.81 µs      │ 21.94 µs      │ 22.06 µs      │ 100     │ 100
   ╰─ Naive                    │               │               │               │         │
      ├─ L1      155.1 µs      │ 168.2 µs      │ 157 µs        │ 157.7 µs      │ 100     │ 100
      ╰─ L2      20.85 µs      │ 24.39 µs      │ 21.24 µs      │ 21.3 µs       │ 100     │ 100
```

<details>

<summary>Command</summary>

```sh
cargo bench --profile release
```

</details>

## Profile


<details>

<summary>Command</summary>

```sh
cargo build --example simple --profile profiling \
 && samply record target/profiling/examples/simple tests/signals-large.txt
```

</details>

## Credits

- [fastpelt](https://github.com/ritchie46/fastpelt)
- [ruptures](https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/)
