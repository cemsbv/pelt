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

| Benchmark | Min (+) | Max (+) | Mean (+) |
| -- | -- | -- | -- |
| ruptures L1 vs pelt L1 | -112.9x |  -114.4x | -113.9x |
| ruptures L2 vs pelt L2 | -298.3x | -304.3x | -301.1x |

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
bench                fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ large                           │               │               │               │         │
│  ├─ Kahan<f64>                   │               │               │               │         │
│  │  ├─ L1          160.1 ms      │ 196.7 ms      │ 171.6 ms      │ 170.8 ms      │ 100     │ 100
│  │  ╰─ L2          13.48 ms      │ 15.92 ms      │ 14.19 ms      │ 14.12 ms      │ 100     │ 100
│  ╰─ Naive<f64>                   │               │               │               │         │
│     ├─ L1          124.3 ms      │ 144.7 ms      │ 132.5 ms      │ 131.4 ms      │ 100     │ 100
│     ╰─ L2          13.56 ms      │ 16.5 ms       │ 14.64 ms      │ 14.59 ms      │ 100     │ 100
╰─ small                           │               │               │               │         │
   ├─ Kahan<f64>                   │               │               │               │         │
   │  ├─ L1          246 µs        │ 268.7 µs      │ 250.8 µs      │ 252.3 µs      │ 100     │ 100
   │  ╰─ L2          48.85 µs      │ 72.17 µs      │ 53.48 µs      │ 54.64 µs      │ 100     │ 100
   ╰─ Naive<f64>                   │               │               │               │         │
      ├─ L1          187.7 µs      │ 229.5 µs      │ 193.3 µs      │ 194.8 µs      │ 100     │ 100
      ╰─ L2          46.99 µs      │ 71.8 µs       │ 51.51 µs      │ 52.54 µs      │ 100     │ 100
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
