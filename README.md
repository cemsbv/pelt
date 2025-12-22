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
| ruptures L1 vs pelt L1 | -115.3x | -118.1x | -116.4x |
| ruptures L2 vs pelt L2 | -347.4x | -365.4x | -353.3x |

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
│  │  ├─ L1      158.8 ms      │ 184.8 ms      │ 159.6 ms      │ 163.6 ms      │ 100     │ 100
│  │  ╰─ L2      7.963 ms      │ 10.35 ms      │ 8.684 ms      │ 8.639 ms      │ 100     │ 100
│  ╰─ Naive                    │               │               │               │         │
│     ├─ L1      122.6 ms      │ 159.7 ms      │ 131.9 ms      │ 132.2 ms      │ 100     │ 100
│     ╰─ L2      7.676 ms      │ 9.949 ms      │ 8.326 ms      │ 8.458 ms      │ 100     │ 100
╰─ small                       │               │               │               │         │
   ├─ Kahan                    │               │               │               │         │
   │  ├─ L1      223.7 µs      │ 300 µs        │ 225.9 µs      │ 232.5 µs      │ 100     │ 100
   │  ╰─ L2      19.06 µs      │ 53.73 µs      │ 19.39 µs      │ 20.7 µs       │ 100     │ 100
   ╰─ Naive                    │               │               │               │         │
      ├─ L1      162.5 µs      │ 240.6 µs      │ 166.3 µs      │ 170.4 µs      │ 100     │ 100
      ╰─ L2      16.89 µs      │ 24.88 µs      │ 17.42 µs      │ 17.64 µs      │ 100     │ 100
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
