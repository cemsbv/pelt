//! L2 cost function.

use std::ops::Range;

use ndarray::{ArrayView1, ArrayView2};

/// Precalculation output.
pub struct L2Cost1D {
    /// Sum query.
    sums: Vec<Sums>,
}

impl L2Cost1D {
    /// Precalculate the sum queries.
    #[inline]
    pub(crate) fn precalculate(signal: &ArrayView1<f64>) -> Self {
        // Calculate the sum of all previous values
        let mut sums = vec![Sums::default(); signal.len()];
        // Sum all previous numbers
        let mut sum_counter = 0.0;
        // Sum the squares of all previous numbers
        let mut sum_squared_counter = 0.0;

        sums.iter_mut()
            .zip(signal.iter())
            .for_each(|(sums, signal)| {
                sum_counter += *signal;
                sum_squared_counter += signal.powi(2);
                sums.sum = sum_counter;
                sums.sum_squared = sum_squared_counter;
            });

        Self { sums }
    }

    /// Calculate the loss.
    ///
    /// Calculated using Welford's algorithm.
    #[inline]
    pub(crate) fn loss(&self, range: Range<usize>) -> f64 {
        // How many rows there are
        let rows_length = range.end.saturating_sub(range.start) as f64;

        // Take the left values or zero if the range is zero
        // We use a wrapping sub for that so when it overflows the get will always return `None`
        let left = self
            .sums
            .get(range.start.wrapping_sub(1))
            .cloned()
            .unwrap_or_default();

        let right = &self.sums[range.end.saturating_sub(1)];

        // Use the sum query to find the sums
        let sum = right.sum - left.sum;
        let sum_squared = right.sum_squared - left.sum_squared;

        // Calculate sum of squares using Welford's algorithm
        sum_squared - sum.powi(2) / rows_length
    }
}

/// Precalculation output.
pub struct L2Cost2D {
    /// Precalculated per column.
    columns: Vec<L2Cost1D>,
}

impl L2Cost2D {
    /// Precalculate the sum queries.
    #[inline]
    pub fn precalculate(signal: &ArrayView2<f64>) -> Self {
        let columns = signal
            .columns()
            .into_iter()
            .map(|column| L2Cost1D::precalculate(&column))
            .collect();

        Self { columns }
    }

    /// Calculate the loss.
    ///
    /// Calculated using Welford's algorithm.
    #[inline]
    pub(crate) fn loss(&self, range: Range<usize>) -> f64 {
        // Calculate total loss
        self.columns
            .iter()
            .map(|column| column.loss(range.clone()))
            .sum()
    }
}

/// All precalculated sum values.
#[derive(Default, Clone)]
struct Sums {
    /// Basic sum.
    sum: f64,
    /// Squared sum.
    sum_squared: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check the L2 cost function.
    #[test]
    fn cost_1d() {
        let array_1d = ndarray::array![10.0, 30.0, 20.0];
        let cost = L2Cost1D::precalculate(&array_1d.view());
        let result = cost.loss(0..3);
        assert_eq!(result, 200.0);
    }

    #[test]
    fn cost_2d() {
        let array_2d = ndarray::array![[10.0], [30.0], [20.0]];
        let cost = L2Cost2D::precalculate(&array_2d.view());
        let result = cost.loss(0..3);
        assert_eq!(result, 200.0);
    }
}
