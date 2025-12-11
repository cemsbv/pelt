//! Changepoint detection with Pruned Exact Linear Time.

mod cost;
mod error;

use std::{iter::Sum, num::NonZero, ops::Range};

use ahash::AHashMap;
pub use cost::SegmentCostFunction;
pub use error::Error;
use ndarray::{ArrayView2, AsArray, Ix2};
use num_traits::{Float, NumCast, float::TotalOrder};

/// PELT algorithm.
///
/// # Defaults
///
/// - `segment_cost_function`: [`SegmentCostFunction::L1`]
/// - `jump`: `5`
/// - `min_length`: `2`
/// - `keep_initial_zero`: `false`
#[derive(Debug, Clone, Copy)]
pub struct Pelt {
    /// Segment model.
    segment_cost_function: SegmentCostFunction,
    /// Subsample, one every `jump` points.
    jump: usize,
    /// Minimum allowable number of data points within a segment.
    min_length: usize,
    /// Whether to keep the initial `0` value of the output indices.
    keep_initial_zero: bool,
}

impl Pelt {
    /// Construct a new PELT instance with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            segment_cost_function: SegmentCostFunction::L1,
            jump: 5,
            min_length: 2,
            keep_initial_zero: false,
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
        self.min_length = minimum_segment_length.get();

        self
    }

    /// Set whether to keep the initial zero value of the output indices.
    #[must_use]
    pub const fn with_keep_initial_zero(mut self, keep_initial_zero: bool) -> Self {
        self.keep_initial_zero = keep_initial_zero;

        self
    }

    /// Fit on a data set.
    ///
    /// # Errors
    ///
    /// - When the input is invalid.
    /// - When anything went wrong during calculation.
    pub fn predict<'a, T>(
        &self,
        signal: impl AsArray<'a, T, Ix2>,
        penalty: T,
    ) -> Result<Vec<usize>, Error>
    where
        T: Float + TotalOrder + NumCast + Sum + 'a,
    {
        let signal_view = signal.into();

        self.predict_impl(signal_view, penalty)
    }

    /// [`Self::predict`] implementation outside of generic to avoid code duplication.
    fn predict_impl<T>(&self, signal: ArrayView2<T>, penalty: T) -> Result<Vec<usize>, Error>
    where
        T: Float + TotalOrder + NumCast + Sum,
    {
        // `partitions[t]` stores the optimal partition of `signal[0..t]`
        let mut partitions: AHashMap<usize, AHashMap<Range<usize>, T>> = AHashMap::new();
        let mut first_partition = AHashMap::new();
        first_partition.insert(Range::default(), T::zero());
        partitions.insert(0, first_partition);

        // List of indices we can accept
        let mut admissible = Vec::with_capacity(self.jump);

        // Pre-allocate it outside of the loop
        let mut subproblems = Vec::with_capacity(self.jump);

        // Find the initial changepoint indices
        for breakpoint in self.proposed_indices(signal.nrows()) {
            // Add points from 0 to the current breakpoint as admissible
            let new_admission_point =
                (breakpoint.saturating_sub(self.min_length) / self.jump) * self.jump;
            admissible.push(new_admission_point);

            // Split admissible into sub problems
            for admissible_start in &admissible {
                // Skip breakpoints without partitions
                let Some(partition) = partitions.get(admissible_start) else {
                    continue;
                };

                // Handle invalid case for too short segments
                if breakpoint.saturating_sub(*admissible_start) < self.min_length {
                    return Err(Error::NotEnoughPoints);
                }

                let range = *admissible_start..breakpoint;

                // Calculate loss function for the admissible range
                let loss = self.segment_cost_function.loss(signal, range.clone());

                // Update with the right partition
                let mut new_partition = partition.clone();
                new_partition.insert(range, loss + penalty);
                subproblems.push(new_partition);
            }

            // Find the optimal partition with the lowest loss
            let mut min_partition = subproblems.first().ok_or(Error::NoSegmentsFound)?;
            let mut min_val = min_partition.values().copied().sum::<T>();
            for (index, subproblem) in subproblems
                .iter()
                .enumerate()
                // Skip the first item since that's the min variables
                .skip(1)
            {
                let sum = subproblem.values().copied().sum::<T>();
                if sum < min_val {
                    min_val = sum;
                    min_partition = &subproblems[index];
                }
            }
            // Assign optimal partition to the map
            partitions.insert(breakpoint, min_partition.clone());

            // Threshold loss to filter each partition
            let loss_current_part = min_val + penalty;

            // Filter the admissible array
            admissible = admissible
                .into_iter()
                // Clear the subproblems array
                .zip(subproblems.drain(..))
                // Keep the admissible parts that follow the loss function
                .filter_map(|(admissible_start, partition)| {
                    (partition.values().copied().sum::<T>() < loss_current_part)
                        .then_some(admissible_start)
                })
                .collect();
        }

        // Get the best partition
        let best_part = partitions
            .get(&signal.nrows())
            .ok_or(Error::NoSegmentsFound)?;

        // Extract the indices
        let mut indices = best_part.keys().map(|range| range.end).collect::<Vec<_>>();

        // Sort indices
        indices.sort_unstable();

        if !self.keep_initial_zero {
            // Remove the zero value
            indices.remove(0);
        }

        Ok(indices)
    }

    /// Calculate the proposed changepoint indices.
    fn proposed_indices(&self, signal_len: usize) -> impl Iterator<Item = usize> {
        // Skip the minimum length to the next jump
        let start = self
            .min_length
            // If it's zero nothing will be skipped
            .saturating_sub(1)
            // Also skip to the next jump position
            .next_multiple_of(self.jump);

        (start..signal_len)
            // Take a index every "jump" items
            .step_by(self.jump)
            // Add the last item
            .chain(std::iter::once(signal_len))
    }
}

impl Default for Pelt {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensure the proposed indices algorithm is correct.
    #[test]
    fn proposed_indices() {
        assert_eq!(
            Pelt::new()
                .with_jump(NonZero::new(5).expect("Invalid number"))
                .with_minimum_segment_length(NonZero::new(2).expect("Invalid number"))
                .proposed_indices(20)
                .collect::<Vec<_>>(),
            vec![5, 10, 15, 20]
        );

        assert_eq!(
            Pelt::new()
                .with_jump(NonZero::new(5).expect("Invalid number"))
                .with_minimum_segment_length(NonZero::new(8).expect("Invalid number"))
                .proposed_indices(20)
                .collect::<Vec<_>>(),
            vec![10, 15, 20]
        );
    }
}
