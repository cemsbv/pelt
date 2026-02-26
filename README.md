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
| _L2_ | _100_ | _1D_ | 2.197 μs | 3.125 ms | **1422.4x** |
| _L2_ | _100_ | _2D_ | 2.478 μs | 3.160 ms | **1275.0x** |
| _L2_ | _1000_ | _1D_ | 126.394 μs | 195.964 ms | **1550.4x** |
| _L2_ | _1000_ | _2D_ | 57.435 μs | 95.860 ms | **1669.0x** |
| _L2_ | _10000_ | _1D_ | 21.401 ms | 12.571 s | 587.4x |
| _L2_ | _10000_ | _2D_ | 2.397 ms | 1.699 s | 708.8x |
| _L1_ | _100_ | _1D_ | 10.613 μs | 4.804 ms | 452.7x |
| _L1_ | _100_ | _2D_ | 21.625 μs | 5.105 ms | 236.1x |
| _L1_ | _1000_ | _1D_ | 321.511 μs | 184.912 ms | 575.1x |
| _L1_ | _1000_ | _2D_ | 2.152 ms | 636.542 ms | 295.9x |
| _L1_ | _10000_ | _1D_ | 13.598 ms | 15.572 s | **1145.2x** |
| _L1_ | _10000_ | _2D_ | 85.147 ms | 30.706 s | 360.6x |

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
