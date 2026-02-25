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
| _L2_ | _100_ | _1D_ | 2.191 μs | 3.051 ms | **1392.8x** |
| _L2_ | _100_ | _2D_ | 2.472 μs | 3.057 ms | **1236.9x** |
| _L2_ | _1000_ | _1D_ | 126.128 μs | 184.303 ms | **1461.2x** |
| _L2_ | _1000_ | _2D_ | 57.869 μs | 94.754 ms | **1637.4x** |
| _L2_ | _10000_ | _1D_ | 21.446 ms | 12.417 s | 579.0x |
| _L2_ | _10000_ | _2D_ | 2.371 ms | 1.731 s | 729.9x |
| _L1_ | _100_ | _1D_ | 10.833 μs | 4.626 ms | 427.0x |
| _L1_ | _100_ | _2D_ | 22.450 μs | 5.197 ms | 231.5x |
| _L1_ | _1000_ | _1D_ | 316.382 μs | 178.432 ms | 564.0x |
| _L1_ | _1000_ | _2D_ | 2.166 ms | 618.521 ms | 285.6x |
| _L1_ | _10000_ | _1D_ | 13.353 ms | 15.554 s | **1164.8x** |
| _L1_ | _10000_ | _2D_ | 87.608 ms | 30.575 s | 349.0x |

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
