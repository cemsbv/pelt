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
| _L2_ | _100_ | _1D_ | 0.004ms | 3.113ms | **753.1x** |
| _L2_ | _100_ | _2D_ | 0.005ms | 3.251ms | **608.5x** |
| _L2_ | _1000_ | _1D_ | 0.419ms | 188.546ms | 449.8x |
| _L2_ | _1000_ | _2D_ | 0.572ms | 94.201ms | 164.7x |
| _L2_ | _10000_ | _1D_ | 95.851ms | 12.103s | 126.3x |
| _L2_ | _10000_ | _2D_ | 19.871ms | 1.778s | 89.5x |
| _L1_ | _100_ | _1D_ | 0.010ms | 5.101ms | **508.8x** |
| _L1_ | _100_ | _2D_ | 0.025ms | 5.323ms | 213.0x |
| _L1_ | _1000_ | _1D_ | 0.641ms | 180.250ms | 281.3x |
| _L1_ | _1000_ | _2D_ | 3.595ms | 638.049ms | 177.5x |
| _L1_ | _10000_ | _1D_ | 43.444ms | 15.710s | 361.6x |
| _L1_ | _10000_ | _2D_ | 364.553ms | 30.572s | 83.9x |

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
