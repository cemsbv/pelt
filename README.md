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

predict(signal, penalty=20, segment_cost_function="l1", jump=10, minimum_segment_length=2, keep_initial_zero=False)
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
let result = pelt.predict::<Kahan<_>, _>(&signal[..], penalty)?;
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
| ruptures L1 vs pelt L1 | -102.1x |  -101.4x | -101.6x |
| ruptures L2 vs pelt L2 | -1578.8x | -1587.7x | -1591.1x |

<details>

<summary>Command</summary>

```sh
richbench benches/
```

</details>

### Rust

```
Timer precision: 20 ns
bench                fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ large                           │               │               │               │         │
│  ├─ Kahan<f64>                   │               │               │               │         │
│  │  ├─ L1          161.2 ms      │ 202.3 ms      │ 162.3 ms      │ 165.3 ms      │ 100     │ 100
│  │  ╰─ L2          4.832 ms      │ 4.923 ms      │ 4.845 ms      │ 4.852 ms      │ 100     │ 100
│  ╰─ Naive<f64>                   │               │               │               │         │
│     ├─ L1          126.6 ms      │ 159.2 ms      │ 127.7 ms      │ 131.5 ms      │ 100     │ 100
│     ╰─ L2          1.436 ms      │ 1.591 ms      │ 1.45 ms       │ 1.455 ms      │ 100     │ 100
╰─ small                           │               │               │               │         │
   ├─ Kahan<f64>                   │               │               │               │         │
   │  ├─ L1          247.4 µs      │ 295.3 µs      │ 252.7 µs      │ 254 µs        │ 100     │ 100
   │  ╰─ L2          65.22 µs      │ 73.42 µs      │ 66.02 µs      │ 66.31 µs      │ 100     │ 100
   ╰─ Naive<f64>                   │               │               │               │         │
      ├─ L1          189.7 µs      │ 254.7 µs      │ 196.3 µs      │ 197.5 µs      │ 100     │ 100
      ╰─ L2          27.19 µs      │ 38.14 µs      │ 28.05 µs      │ 28.45 µs      │ 100     │ 100
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
