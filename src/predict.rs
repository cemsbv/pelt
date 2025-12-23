//! Predict implementation.

use std::ops::Range;

use ahash::AHashMap;
use fearless_simd::Level;
use ndarray::ArrayView2;
use smallvec::SmallVec;

use crate::{Error, Pelt, Sum};

/// Implementation of predict with state.
pub struct PredictImpl<S> {
    /// Pelt data.
    pelt: Pelt,
    /// `partitions[t]` stores the optimal partition of `signal[0..t]`.
    partitions: AHashMap<usize, Partition<S>>,
    /// List of indices we can accept
    admissible: Vec<usize>,
    /// All subproblems.
    subproblems: Vec<Partition<S>>,
    /// What SIMD features we can use.
    simd_level: Level,
}

impl<S: Sum<f64> + Send + Sync> PredictImpl<S> {
    /// Setup the structures.
    pub(crate) fn new(pelt: Pelt) -> Self {
        // Detect the SIMD mechanism at runtime
        let simd_level = Level::new();

        // `partitions[t]` stores the optimal partition of `signal[0..t]`
        let mut partitions = AHashMap::new();
        partitions.insert(0, Partition::default());

        // List of indices we can accept
        let admissible = Vec::with_capacity(pelt.jump);

        // Pre-allocate it outside of the loop
        let subproblems = Vec::with_capacity(pelt.jump);

        Self {
            pelt,
            partitions,
            admissible,
            subproblems,
            simd_level,
        }
    }

    /// Run the calculation loop.
    pub(crate) fn predict(
        &mut self,
        signal: ArrayView2<f64>,
        penalty: f64,
    ) -> Result<Vec<usize>, Error> {
        // Find the initial changepoint indices
        for breakpoint in self.proposed_indices(signal.nrows()) {
            // Add points from 0 to the current breakpoint as admissible
            let new_admission_point = (breakpoint.saturating_sub(self.pelt.minimum_segment_length)
                / self.pelt.jump)
                * self.pelt.jump;
            self.admissible.push(new_admission_point);

            // Reset subproblems
            self.subproblems.clear();

            // Split admissible into sub problems based on a heuristic
            // The heuristic determines whether the overhead of starting the threads is worth it
            #[cfg(feature = "rayon")]
            if self
                .pelt
                .segment_cost_function
                .should_use_threading(self.admissible.len())
            {
                // Use all available threads
                self.par_split_into_subproblems(breakpoint, signal, penalty)?;
            } else {
                // Keep using a single thread
                self.split_into_subproblems(breakpoint, signal, penalty)?;
            }

            // Split admissible into sub problems
            #[cfg(not(feature = "rayon"))]
            self.split_into_subproblems(breakpoint, signal, penalty)?;

            // Find the optimal partition with the lowest loss
            let min_subproblem = self
                .subproblems
                .iter()
                .min_by(|left, right| {
                    left.loss_and_penalty_sum()
                        .total_cmp(&right.loss_and_penalty_sum())
                })
                .ok_or(Error::NotEnoughPoints)?;

            // Assign optimal partition to the map
            self.partitions.insert(breakpoint, min_subproblem.clone());

            // Threshold loss to filter each partition
            let loss_current_part = min_subproblem.loss_and_penalty_sum() + penalty;

            // We apply a zip to the subproblems manually
            self.admissible.resize(self.subproblems.len(), 0);

            // Filter the admissible array
            let mut index = 0;
            self.admissible.retain(|_admissible| {
                // Drain and zip the subproblems
                let subproblem = &self.subproblems[index];
                index += 1;

                subproblem.loss_and_penalty_sum() < loss_current_part
            });
        }

        // Get the best partition
        let best_part = self
            .partitions
            .remove(&signal.nrows())
            .ok_or(Error::NoSegmentsFound)?;

        // Extract the indices
        let mut indices = best_part.ranges;

        // Sort indices
        indices.sort_unstable();

        Ok(indices.to_vec())
    }

    /// Calculate the proposed changepoint indices.
    #[inline]
    fn proposed_indices(&self, signal_len: usize) -> impl Iterator<Item = usize> + use<S> {
        // Skip the minimum length to the next jump
        let start = self
            .pelt
            .minimum_segment_length
            // If it's zero nothing will be skipped
            .saturating_sub(1)
            // Also skip to the next jump position
            .next_multiple_of(self.pelt.jump);

        (start..signal_len)
            // Take a index every "jump" items
            .step_by(self.pelt.jump)
            // Add the last item
            .chain(std::iter::once(signal_len))
    }

    /// Split admissible into sub problems based on the breakpoint.
    #[inline]
    fn split_into_subproblems(
        &mut self,
        breakpoint: usize,
        signal: ArrayView2<f64>,
        penalty: f64,
    ) -> Result<(), Error> {
        for admissible_start in &self.admissible {
            self.subproblems.push(self.find_subproblem(
                signal,
                *admissible_start..breakpoint,
                penalty,
            )?);
        }

        Ok(())
    }

    /// Split admissible into sub problems based on the breakpoint, spread across threads.
    #[cfg(feature = "rayon")]
    #[inline]
    fn par_split_into_subproblems(
        &mut self,
        breakpoint: usize,
        signal: ArrayView2<f64>,
        penalty: f64,
    ) -> Result<(), Error> {
        use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};

        self.subproblems = self
            .admissible
            .par_iter()
            .map(|admissible_start| {
                self.find_subproblem(signal, *admissible_start..breakpoint, penalty)
            })
            .collect::<Result<_, _>>()?;

        Ok(())
    }

    /// Calculate loss for a specific partition and add it to a new partition.
    #[inline]
    fn find_subproblem(
        &self,
        signal: ArrayView2<f64>,
        range: Range<usize>,
        penalty: f64,
    ) -> Result<Partition<S>, Error> {
        // Handle case where there's no partitions yet, shouldn't happen
        let Some(partition) = self.partitions.get(&range.start) else {
            branches::mark_unlikely();
            return Err(Error::NotEnoughPoints);
        };

        // Handle invalid case for too short segments
        if branches::unlikely(
            range.end.saturating_sub(range.start) < self.pelt.minimum_segment_length,
        ) {
            return Err(Error::NotEnoughPoints);
        }

        // Calculate loss function for the admissible range
        let loss = self
            .pelt
            .segment_cost_function
            .loss(self.simd_level, signal, range.clone());

        // Update with the right partition
        let mut new_partition = partition.clone();
        new_partition.push(range.end, loss, penalty);

        Ok(new_partition)
    }
}

/// A single partition.
#[derive(Clone)]
struct Partition<S> {
    /// End of ranges it applies to.
    ranges: SmallVec<usize, 8>,
    /// Sum of all loss and penalty values.
    loss_and_penalty_sum: S,
}

impl<S> Partition<S>
where
    S: Sum<f64>,
{
    /// Push a new value.
    #[inline]
    pub fn push(&mut self, range: usize, loss: S, penalty: f64) {
        self.ranges.push(range);

        self.loss_and_penalty_sum = self.loss_and_penalty_sum.clone() + loss.sum() + penalty;
    }

    /// Get the sum of the loss and penalty.
    #[inline]
    pub fn loss_and_penalty_sum(&self) -> f64 {
        self.loss_and_penalty_sum.clone().sum()
    }
}

impl<S> Default for Partition<S>
where
    S: Sum<f64>,
{
    #[inline]
    fn default() -> Self {
        Self {
            ranges: SmallVec::new(),
            loss_and_penalty_sum: S::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use crate::Kahan;

    use super::*;

    /// Ensure the proposed indices algorithm is correct.
    #[test]
    fn proposed_indices() {
        assert_eq!(
            PredictImpl::<Kahan>::new(
                Pelt::new()
                    .with_jump(NonZero::new(5).expect("Invalid number"))
                    .with_minimum_segment_length(NonZero::new(2).expect("Invalid number"))
            )
            .proposed_indices(20)
            .collect::<Vec<_>>(),
            vec![5, 10, 15, 20]
        );

        assert_eq!(
            PredictImpl::<Kahan>::new(
                Pelt::new()
                    .with_jump(NonZero::new(5).expect("Invalid number"))
                    .with_minimum_segment_length(NonZero::new(8).expect("Invalid number"))
            )
            .proposed_indices(20)
            .collect::<Vec<_>>(),
            vec![10, 15, 20]
        );
    }
}
