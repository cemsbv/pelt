//! L1 cost functions.

use std::ops::Range;

use ndarray::{ArrayView1, ArrayView2};

use crate::cost::tree::KthSmallestTree;

/// Precalculation output.
pub struct L1Cost1D {
    /// Tree for finding the mean (middle K-th smallest).
    kth_smallest_tree: KthSmallestTree,
}

impl L1Cost1D {
    /// Build the K-th smallest tree.
    #[inline]
    pub(crate) fn precalculate(signal: &ArrayView1<f64>) -> Self {
        // Build the tree from the signal
        let kth_smallest_tree = KthSmallestTree::build(signal);

        Self { kth_smallest_tree }
    }

    /// Calculate the loss.
    #[inline]
    pub(crate) fn loss(&self, signal: &ArrayView1<f64>, range: Range<usize>) -> f64 {
        // Calculate the median for the segment
        let median = self.median(range.clone());

        // Take the sub slice of the 2D object
        signal
            .slice(ndarray::s!(range))
            // Calculate the absolute difference for each point with the median
            .iter()
            .map(|signal| (*signal - median).abs())
            // Sum all values
            .sum()
    }

    /// Get the median of a range in the signal.
    #[inline]
    fn median(&self, range: Range<usize>) -> f64 {
        // Find the midpoint of the range
        let len = range.len();

        // Convert the range to an inclusive one
        let range_inclusive = range.start..=(range.end - 1);

        // Calculate median based one one or two variables if it's even
        if len.is_multiple_of(2) {
            // Get the middle value, offset by 1 for the two points
            let kth = len / 2;

            // Get the average of the two median
            let median1 = self.kth_smallest_tree.kth(range_inclusive.clone(), kth);
            let median2 = self.kth_smallest_tree.kth(range_inclusive, kth + 1);

            median1.midpoint(median2)
        } else {
            // Get the middle value
            let kth = len / 2 + 1;

            // Get the median
            self.kth_smallest_tree.kth(range_inclusive, kth)
        }
    }
}

/// Precalculation output.
pub struct L1Cost2D {
    /// Precalculated per column.
    columns: Vec<L1Cost1D>,
}

impl L1Cost2D {
    /// Precalculate the sum queries.
    #[inline]
    pub fn precalculate(signal: &ArrayView2<f64>) -> Self {
        let columns = signal
            .columns()
            .into_iter()
            .map(|column| L1Cost1D::precalculate(&column))
            .collect();

        Self { columns }
    }

    /// Calculate the loss.
    ///
    /// Calculated using Welford's algorithm.
    #[inline]
    pub(crate) fn loss(&self, signal: &ArrayView2<f64>, range: Range<usize>) -> f64 {
        // Calculate total loss
        self.columns
            .iter()
            .zip(signal.columns())
            .map(|(column, signal_column)| column.loss(&signal_column, range.clone()))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check the L1 cost function.
    #[test]
    fn cost_1d() {
        let array_1d = ndarray::array![10.0, 30.0, 20.0];
        let cost = L1Cost1D::precalculate(&array_1d.view());
        let result = cost.loss(&array_1d.view(), 0..3);
        assert_eq!(result, 20.0);
    }

    /// Check the L1 cost function.
    #[test]
    fn cost_2d() {
        let array_2d = ndarray::array![[10.0], [30.0], [20.0]];
        let cost = L1Cost2D::precalculate(&array_2d.view());
        let result = cost.loss(&array_2d.view(), 0..3);
        assert_eq!(result, 20.0);
    }
}
