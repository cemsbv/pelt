//! Changepoint detection with Pruned Exact Linear Time.

mod cost;
mod error;
#[cfg(feature = "python")]
mod python;

use std::{marker::PhantomData, num::NonZero};

use ahash::AHashMap;
pub use cost::SegmentCostFunction;
pub use error::Error;
use ndarray::{ArrayView2, AsArray, Ix2};
use num_traits::{Float, Zero, float::TotalOrder};

/// Kahan summation
///
/// Source: [`accurate::sum::Kahan`], slower but more accurate.
pub type Kahan<T> = accurate::sum::Kahan<T>;

/// Naive floating point summation, very fast but inaccurate.
///
/// Source: [`accurate::sum::NaiveSum`].
pub type Naive<T> = accurate::sum::NaiveSum<T>;

/// Which sum algorithm to use.
pub use accurate::traits::SumAccumulator as Sum;

/// PELT algorithm.
///
/// # Defaults
///
/// - `segment_cost_function`: [`SegmentCostFunction::L1`]
/// - `jump`: `5`
/// - `minimum_segment_length`: `2`
/// - `keep_initial_zero`: `false`
#[derive(Debug, Clone, Copy)]
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
    pub fn predict<'a, S, T>(
        &self,
        signal: impl AsArray<'a, T, Ix2>,
        penalty: T,
    ) -> Result<Vec<usize>, Error>
    where
        S: Sum<T>,
        T: Float + TotalOrder + 'a,
    {
        let signal_view = signal.into();

        self.predict_impl::<S, T>(signal_view, penalty)
    }

    /// [`Self::predict`] implementation outside of generic to avoid code duplication.
    fn predict_impl<S, T>(&self, signal: ArrayView2<T>, penalty: T) -> Result<Vec<usize>, Error>
    where
        S: Sum<T>,
        T: Float + TotalOrder,
    {
        // `partitions[t]` stores the optimal partition of `signal[0..t]`
        let mut partitions: AHashMap<usize, Partition<S, T>> = AHashMap::new();
        partitions.insert(0, Partition::default());

        // List of indices we can accept
        let mut admissible = Vec::with_capacity(self.jump);

        // Pre-allocate it outside of the loop
        let mut subproblems = Vec::with_capacity(self.jump);

        // Find the initial changepoint indices
        for breakpoint in self.proposed_indices(signal.nrows()) {
            // Add points from 0 to the current breakpoint as admissible
            let new_admission_point =
                (breakpoint.saturating_sub(self.minimum_segment_length) / self.jump) * self.jump;
            admissible.push(new_admission_point);

            // Split admissible into sub problems
            for admissible_start in &admissible {
                // Handle case where there's no partitions yet, shouldn't happen
                let Some(partition) = partitions.get(admissible_start) else {
                    return Err(Error::NotEnoughPoints);
                };

                // Handle invalid case for too short segments
                if breakpoint.saturating_sub(*admissible_start) < self.minimum_segment_length {
                    return Err(Error::NotEnoughPoints);
                }

                // Calculate loss function for the admissible range
                let loss = self
                    .segment_cost_function
                    .loss(signal, *admissible_start..breakpoint);

                // Update with the right partition
                let mut new_partition = partition.clone();
                new_partition.push(breakpoint, loss, penalty);
                subproblems.push(new_partition);
            }

            // Find the optimal partition with the lowest loss
            let min_subproblem = subproblems
                .iter()
                .min_by(|left, right| {
                    left.loss_and_penalty_sum()
                        .total_cmp(&right.loss_and_penalty_sum())
                })
                .ok_or(Error::NotEnoughPoints)?;

            // Assign optimal partition to the map
            partitions.insert(breakpoint, min_subproblem.clone());

            // Threshold loss to filter each partition
            let loss_current_part = min_subproblem.loss_and_penalty_sum() + penalty;

            // Filter the admissible array
            admissible = admissible
                .into_iter()
                // Clear the subproblems array
                .zip(subproblems.drain(..))
                // Keep the admissible parts that follow the loss function
                .filter_map(|(admissible_start, subproblem)| {
                    (subproblem.loss_and_penalty_sum() < loss_current_part)
                        .then_some(admissible_start)
                })
                .collect();
        }

        // Get the best partition
        let best_part = partitions
            .remove(&signal.nrows())
            .ok_or(Error::NoSegmentsFound)?;

        // Extract the indices
        let mut indices = best_part.ranges;

        // Sort indices
        indices.sort_unstable();

        Ok(indices.to_vec())
    }

    /// Calculate the proposed changepoint indices.
    fn proposed_indices(&self, signal_len: usize) -> impl Iterator<Item = usize> {
        // Skip the minimum length to the next jump
        let start = self
            .minimum_segment_length
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

/// A single partition.
#[derive(Clone)]
struct Partition<S, T> {
    /// End of ranges it applies to.
    ranges: Vec<usize>,
    /// Sum of all loss and penalty values.
    loss_and_penalty_sum: S,
    /// Ignore `T`.
    _phantom: PhantomData<T>,
}

impl<S, T> Partition<S, T>
where
    S: Sum<T>,
{
    /// Push a new value.
    #[inline]
    pub fn push(&mut self, range: usize, loss: S, penalty: T) {
        self.ranges.push(range);

        self.loss_and_penalty_sum = self.loss_and_penalty_sum.clone() + loss.sum() + penalty;
    }

    /// Get the sum of the loss and penalty.
    #[inline]
    pub fn loss_and_penalty_sum(&self) -> T {
        self.loss_and_penalty_sum.clone().sum()
    }
}

impl<S, T> Default for Partition<S, T>
where
    S: Sum<T>,
    T: Zero,
{
    #[inline]
    fn default() -> Self {
        Self {
            ranges: Vec::with_capacity(8),
            loss_and_penalty_sum: S::zero(),
            _phantom: PhantomData,
        }
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
