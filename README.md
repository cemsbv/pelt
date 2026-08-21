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

predict(signal, penalty=20, segment_cost_function="l1", jump=10, minimum_segment_length=2)
```

### Rust

```rust
use std::num::NonZero;
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
maturin develop

# Open an interpreter
python

>>> from pelt import predict
>>> import numpy as np
>>> signal = np.array([np.sin(np.arange(0, 1000, 10))]).transpose()
>>> predict(signal, penalty=20)
```

## Benchmarks

> [!WARNING]
> Like all benchmarks, take these with a grain of salt.

### Python

Comparison with [ruptures](https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/):

| Cost Function | Data Points | Data Dimension | Mean `pelt` | Mean `ruptures` | Times Faster |
| -- | -- | -- | -- | -- | -- |
| _L2_ | _100_ | _1D_ | 2.065 μs | 2.820 ms | **1365.7x** |
| _L2_ | _100_ | _2D_ | 2.294 μs | 2.817 ms | **1228.1x** |
| _L2_ | _1000_ | _1D_ | 107.256 μs | 171.377 ms | **1597.8x** |
| _L2_ | _1000_ | _2D_ | 51.096 μs | 90.582 ms | **1772.8x** |
| _L2_ | _10000_ | _1D_ | 20.038 ms | 11.454 s | 571.6x |
| _L2_ | _10000_ | _2D_ | 2.751 ms | 1.672 s | 607.7x |
| _L1_ | _100_ | _1D_ | 9.148 μs | 4.312 ms | 471.3x |
| _L1_ | _100_ | _2D_ | 19.886 μs | 4.774 ms | 240.1x |
| _L1_ | _1000_ | _1D_ | 173.917 μs | 162.959 ms | 937.0x |
| _L1_ | _1000_ | _2D_ | 2.269 ms | 588.634 ms | 259.5x |
| _L1_ | _10000_ | _1D_ | 3.515 ms | 14.674 s | **4174.3x** |
| _L1_ | _10000_ | _2D_ | 104.050 ms | 30.726 s | 295.3x |

<details>

<summary>Command</summary>

```sh
maturin develop
python benches/bench_compare.py
```

</details>

## Profile

<details>

<summary>Command</summary>

```sh
cargo build --example simple --profile profiling \
 && samply record target/profiling/examples/simple tests/signals-large.csv
```

</details>

## Credits

- [fastpelt](https://github.com/ritchie46/fastpelt)
- [ruptures](https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/)
