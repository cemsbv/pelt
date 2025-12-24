//! Changepoint detection with Pruned Exact Linear Time.

pub(crate) mod cost;
pub(crate) mod error;
pub(crate) mod predict;
#[cfg(feature = "python")]
mod python;

use std::num::NonZero;

pub use cost::SegmentCostFunction;
pub use error::Error;
use ndarray::{AsArray, Ix2};
use predict::PredictImpl;

/// PELT algorithm.
///
/// # Defaults
///
/// - `segment_cost_function`: [`SegmentCostFunction::L1`]
/// - `jump`: `5`
/// - `minimum_segment_length`: `2`
/// - `keep_initial_zero`: `false`
#[derive(Debug, Clone)]
pub struct Pelt {
    /// Segment model.
    segment_cost_function: SegmentCostFunction,
    /// Subsample, one every `jump` points.
    jump: usize,
    /// Minimum allowable number of data points within a segment.
    minimum_segment_length: usize,
}

impl Pelt {
    /// Construct a new PELT instance with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            segment_cost_function: SegmentCostFunction::L1,
            jump: 5,
            minimum_segment_length: 2,
        }
    }

    /// Set the segment model, also known as the loss function.
    ///
    /// Determines how the cost of each potential segment is calculated.
    #[must_use]
    pub const fn with_segment_cost_function(mut self, model: SegmentCostFunction) -> Self {
        self.segment_cost_function = model;

        self
    }

    /// Set the step size when considering previous potential change points.
    ///
    /// - If `jump = 1`, a check is done every possible prior change point, guaranteeing an exact solution, finding the true minimum of the objective function.
    /// - If `jump > 1`, previous change points are considered at intervals of `jump`. This speeds up the computation, but the solution becomes approximate.
    #[must_use]
    pub const fn with_jump(mut self, jump: NonZero<usize>) -> Self {
        self.jump = jump.get();

        self
    }

    /// Set the minimum allowable number of data points within a segment.
    ///
    /// Ensures that segments are not too small.
    #[must_use]
    pub const fn with_minimum_segment_length(
        mut self,
        minimum_segment_length: NonZero<usize>,
    ) -> Self {
        self.minimum_segment_length = minimum_segment_length.get();

        self
    }

    /// Fit on a data set.
    ///
    /// # Errors
    ///
    /// - When the input is invalid.
    /// - When anything went wrong during calculation.
    pub fn predict<'a>(
        &self,
        signal: impl AsArray<'a, f64, Ix2>,
        penalty: f64,
    ) -> Result<Vec<usize>, Error> {
        let signal_view = signal.into();

        // Using this struct has the additional benefit of reducing code duplication from the signal
        PredictImpl::new(self.clone()).predict(signal_view, penalty)
    }
}

impl Default for Pelt {
    fn default() -> Self {
        Self::new()
    }
}
