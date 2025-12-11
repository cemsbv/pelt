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
use pelt::{Pelt, SegmentCostFunction};

// Setup the structure for calculating changepoints
let pelt = Pelt::new()
  .with_jump(NonZero::new(5).expect("Invalid number"))
  .with_minimum_segment_length(NonZero::new(2).expect("Invalid number"))
  .with_segment_cost_function(SegmentCostFunction::L1);

// Do the calculation on a data set
let penalty = 10.0;
let result = pelt.predict(&signal[..], penalty)?;
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

### Python

Comparison with [ruptures](https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/):

| Benchmark | Min | Max | Mean | Min (+) | Max (+) | Mean (+) |
| -- | -- | -- | -- | -- | -- | -- |
| ruptures L1 vs pelt L1 | 0.124 | 0.127 | 0.125 | 18.050 (-145.4x) | 19.298 (-151.9x) | 18.401 (-147.2x) |
| ruptures L2 vs pelt L2 | 0.099 | 0.099 | 0.099 | 9.317 (-94.2x) | 9.667 (-97.5x) | 9.513 (-96.0x) |

<details>

<summary>Command</summary>

```sh
richbench benches/
```

</details>

### Rust

```
Timer precision: 20 ns
bench     fastest       | slowest       | median        | mean          | samples | iters
├─ large                |               |               |               |         |
|  ├─ L1  99.48 ms      | 133.5 ms      | 107.7 ms      | 109.1 ms      | 100     | 100
|  ╰─ L2  31.42 ms      | 41.47 ms      | 33.18 ms      | 33.96 ms      | 100     | 100
╰─ small                |               |               |               |         |
   ├─ L1  199.8 µs      | 229 µs        | 207.2 µs      | 208 µs        | 100     | 100
   ╰─ L2  56.6 µs       | 69.11 µs      | 57.96 µs      | 58.29 µs      | 100     | 100
```

<details>

<summary>Command</summary>

```sh
cargo bench --profile release
```

</details>

## Credits

- [fastpelt](https://github.com/ritchie46/fastpelt)
- [ruptures](https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/)
