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
maturin develop --features python

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
| _L2_ | _100_ | _1D_ | 2.177 μs | 2.985 ms | **1371.4x** |
| _L2_ | _100_ | _2D_ | 2.624 μs | 3.177 ms | **1210.9x** |
| _L2_ | _1000_ | _1D_ | 137.117 μs | 188.953 ms | **1378.0x** |
| _L2_ | _1000_ | _2D_ | 58.199 μs | 96.658 ms | **1660.8x** |
| _L2_ | _10000_ | _1D_ | 21.454 ms | 12.466 s | 581.1x |
| _L2_ | _10000_ | _2D_ | 2.425 ms | 1.792 s | 738.9x |
| _L1_ | _100_ | _1D_ | 10.239 μs | 4.915 ms | 480.1x |
| _L1_ | _100_ | _2D_ | 27.011 μs | 4.991 ms | 184.8x |
| _L1_ | _1000_ | _1D_ | 653.597 μs | 178.593 ms | 273.2x |
| _L1_ | _1000_ | _2D_ | 3.965 ms | 634.811 ms | 160.1x |
| _L1_ | _10000_ | _1D_ | 46.181 ms | 15.646 s | 338.8x |
| _L1_ | _10000_ | _2D_ | 415.895 ms | 31.007 s | 74.6x |

<details>

<summary>Command</summary>

```sh
maturin develop --features python --release
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
