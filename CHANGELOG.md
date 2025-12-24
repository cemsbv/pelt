# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2025-12-24

### Documentation
- *Readme*: Fix Python example

### Features
- *Rayon*: `rayon` feature flag which runs algorithm in parallel

### Performance
- *Alloc*: Reduce allocations
- *Cost*: Move slice calculation outside of loop
- *Predict*: Reduce allocations by extending from iterator
- *Simd*:  [**BREAKING**]Make all types `f64` to apply SIMD subroutines

### Refactor
- *Dimensions*:  [**BREAKING**]Allow input to be 1D or 2D
- *Math*:  [**BREAKING**]Remove generic summing method because it was not applied consistently

### Testing
- *Ruptures*: Add test from ruptures for L1 and L2

### Bench
- *Cases*: Benchmark 2 dimensional case
- *Python*: Simplify benchmarks
- *Rust*: Move to bencher and criterion

## [0.2.0] - 2025-12-15

### Bug Fixes
- *L2*: Use proper range for calculating rows
- *Math*: Improve numerical stability by using Kahan accumulators for summing

### Documentation
- *Readme*: Show benchmarks

### Features
- *Sum*:  [**BREAKING**]Add generic parameter for choosing sum accuracy/speed

### Performance
- *Algorithm*:  [**BREAKING**]Reduce lookups by using vectors instead of hashmaps

## [0.1.0] - 2025-12-11

### Bug Fixes
- *Predict*: Handle more than 1 dimension properly
- *Python*: Add missing pyproject.toml

### Features
- *Float*: Generalize all input over all float types
- *Options*: `keep_initial_zero` for not removing the zero from the output indices
- *Python*: Create Python bindings

### Miscellaneous Tasks
- *Builder*: Make `Pelt::new` const
- *Clippy*: Enforce and apply additional lints
- *Python*: Rename job
- *Release*: Add pipeline for releasing to PyPi
- *Renovate*: Automerge dependency updates
- *Repo*: Initial commit

### Performance
- *Bench*: Compile optimized binary for benchmarks

- *L1*:
    - Use faster custom median algorithm with a single allocation
    - Use faster median algorithm for L1 cost

### Refactor
- *Builder*:  [**BREAKING**]Make types `NonZero` when value can't be zero
- *Cost*: Split into more clear functions

### Styling
- *Toml*: Add taplo config

### Testing
- *Integration*: Move integration tests to `tests/` folder

### Bench
- *Large*: Use the default sample count

<!-- CEMS BV. -->
