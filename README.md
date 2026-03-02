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
| _L2_ | _100_ | _1D_ | 2.042 μs | 3.110 ms | **1523.1x** |
| _L2_ | _100_ | _2D_ | 2.420 μs | 3.148 ms | **1300.9x** |
| _L2_ | _1000_ | _1D_ | 106.783 μs | 190.108 ms | **1780.3x** |
| _L2_ | _1000_ | _2D_ | 55.815 μs | 104.357 ms | **1869.7x** |
| _L2_ | _10000_ | _1D_ | 20.407 ms | 12.859 s | 630.2x |
| _L2_ | _10000_ | _2D_ | 2.317 ms | 1.797 s | 775.6x |
| _L1_ | _100_ | _1D_ | 11.245 μs | 5.041 ms | 448.3x |
| _L1_ | _100_ | _2D_ | 21.587 μs | 5.350 ms | 247.8x |
| _L1_ | _1000_ | _1D_ | 321.619 μs | 187.656 ms | 583.5x |
| _L1_ | _1000_ | _2D_ | 2.186 ms | 628.126 ms | 287.3x |
| _L1_ | _10000_ | _1D_ | 13.215 ms | 15.615 s | **1181.6x** |
| _L1_ | _10000_ | _2D_ | 84.453 ms | 30.508 s | 361.2x |

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
