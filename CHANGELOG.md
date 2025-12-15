# Changelog

All notable changes to this project will be documented in this file.

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
