//! Cost functions.

use std::ops::Range;

use fearless_simd::{Level, Simd, SimdBase as _, SimdInto as _};
use ndarray::{ArrayView, ArrayView1, ArrayView2, Dimension};

use crate::OneOrTwoDimensions;

/// Segment model cost function, also known as the loss function.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SegmentCostFunction {
    /// Least absolute deviation.
    #[default]
    L1,
    /// Least squared deviation.
    L2,
}

impl SegmentCostFunction {
    /// Calculate the loss.
    #[doc(hidden)]
    #[inline]
    pub fn loss<D>(self, simd_level: Level, signal: &ArrayView<f64, D>, range: Range<usize>) -> f64
    where
        D: OneOrTwoDimensions + Dimension,
    {
        match self {
            Self::L1 => D::l1(signal, range),
            Self::L2 => D::l2(simd_level, signal, range),
        }
    }

    /// Heuristic for determining whether to use a parallel iterator.
    #[cfg(feature = "rayon")]
    #[inline]
    pub(crate) const fn should_use_threading(self, iterations: usize) -> bool {
        match self {
            // L1 is slow, so with a couple of iterations it already pays of
            Self::L1 => iterations >= 32,
            // L2 is quite fast, so it's only worthwhile with many iterations
            Self::L2 => iterations >= 512,
        }
    }
}

/// L1 loss function for 1 dimensional array.
#[inline]
pub(crate) fn l1_1d(signal: &ArrayView1<f64>, range: Range<usize>) -> f64 {
    // Take the sub slice of the 2D object
    let segment = signal.slice(ndarray::s!(range));

    // Calculate the median
    let median = median(segment);

    segment.iter().map(|signal| (*signal - median).abs()).sum()
}

/// L1 loss function for 2 dimensional array.
#[inline]
pub(crate) fn l1_2d(signal: &ArrayView2<f64>, range: Range<usize>) -> f64 {
    // Total loss across all axes
    signal
        .columns()
        .into_iter()
        .map(|column| l1_1d(&column, range.clone()))
        .sum()
}

/// L2 loss function.
///
/// Calculated using Welford's algorithm.
#[inline]
pub(crate) fn l2_1d(simd_level: Level, signal: &ArrayView1<f64>, range: Range<usize>) -> f64 {
    // How many rows there are
    let rows_length = range.end.saturating_sub(range.start) as f64;

    // Take the sub slice of the 2D object
    let segment = signal.slice(ndarray::s!(range));

    // Handle the fast case where we can treat the data as a contiguous slice
    let (sum, sum_sqr) = segment.as_slice().map_or_else(
        || {
            // Slow case, use the sub-optimal non-contiguous iterator
            let mut sum = 0.0;
            let mut sum_sqr = 0.0;

            segment.iter().for_each(|value| {
                sum += *value;
                sum_sqr += value.powi(2);
            });

            (sum, sum_sqr)
        },
        // Fast case, handle with SIMD
        |slice| fearless_simd::dispatch!(simd_level, simd => sum_and_sum_sqr(simd, slice)),
    );

    // Calculate sum of squares using Welford's algorithm
    sum_sqr - sum.powi(2) / rows_length
}

/// L2 loss function.
///
/// Calculated using Welford's algorithm.
#[inline]
pub(crate) fn l2_2d(simd_level: Level, signal: &ArrayView2<f64>, range: Range<usize>) -> f64 {
    // Calculate total loss
    signal
        .columns()
        .into_iter()
        .map(|column| l2_1d(simd_level, &column, range.clone()))
        .sum()
}

/// SIMD dispatch for calculating a sum and a square of sums.
#[inline]
fn sum_and_sum_sqr<S: Simd>(simd: S, slice: &[f64]) -> (f64, f64) {
    // Process in SIMD chunks
    let mut simd_sum: S::f64s = 0.0.simd_into(simd);
    let mut simd_sum_sqr: S::f64s = 0.0.simd_into(simd);
    slice.chunks_exact(S::f64s::N).for_each(|chunk| {
        let values_sum = S::f64s::from_slice(simd, chunk);

        simd_sum += values_sum;
        simd_sum_sqr += values_sum * values_sum;
    });
    let mut sum = simd_sum.as_slice().iter().sum::<f64>();
    let mut sum_sqr = simd_sum_sqr.as_slice().iter().sum::<f64>();

    // Process the remainder
    slice
        .chunks_exact(S::f64s::N)
        .remainder()
        .iter()
        .for_each(|value| {
            sum += *value;
            sum_sqr += value.powi(2);
        });

    (sum, sum_sqr)
}

/// Fast median calculation.
#[inline]
fn median(array: ArrayView1<f64>) -> f64 {
    let len = array.len();

    // Check the easy cases
    match len {
        0 => return 0.0,
        1 => return array[0],
        2 => return array[0].midpoint(array[1]),
        _ => (),
    }

    let mut array = array.to_vec();

    // Handle the case of even and odd arrays
    if len.is_multiple_of(2) {
        // Take the two middle values
        let (_, left, rest) = array.select_nth_unstable_by(len / 2 - 1, f64::total_cmp);
        let (_, right, _) = rest.select_nth_unstable_by(0, f64::total_cmp);

        left.midpoint(*right)
    } else {
        // Take the single midpoint value
        let (_, mid, _) = array.select_nth_unstable_by(len / 2, f64::total_cmp);

        *mid
    }
}

#[cfg(test)]
mod tests {
    use fearless_simd::Level;

    /// Check the L1 cost function.
    #[test]
    fn l1() {
        let array_1d = ndarray::array![10.0, 30.0, 20.0];
        assert_eq!(super::l1_1d(&array_1d.view(), 0..3), 20.0);

        let array_2d = ndarray::array![[10.0], [30.0], [20.0]];
        assert_eq!(super::l1_2d(&array_2d.view(), 0..3), 20.0);
    }

    /// Check the L2 cost function.
    #[test]
    fn l2() {
        let array_1d = ndarray::array![10.0, 30.0, 20.0];
        let result = super::l2_1d(Level::new(), &array_1d.view(), 0..3);
        assert_eq!(result, 200.0);

        let array_2d = ndarray::array![[10.0], [30.0], [20.0]];
        let result = super::l2_2d(Level::new(), &array_2d.view(), 0..3);
        assert_eq!(result, 200.0);
    }

    /// Check the median function.
    #[test]
    fn median() {
        let array = ndarray::array![10.0, 30.0, 20.0];
        assert_eq!(super::median(array.view()), 20.0);
    }
}
