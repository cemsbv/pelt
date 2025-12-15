//! Cost functions.

use std::ops::Range;

use accurate::traits::{SumAccumulator, SumWithAccumulator};
use ndarray::{ArrayView1, ArrayView2};
use num_traits::{Float, float::TotalOrder};

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
    #[inline]
    pub(crate) fn loss<S, T>(self, signal: ArrayView2<T>, range: Range<usize>) -> S
    where
        S: SumAccumulator<T>,
        T: Float + TotalOrder,
    {
        match self {
            Self::L1 => l1(signal, range),
            Self::L2 => l2(signal, range),
        }
    }
}

/// L1 loss function.
#[inline]
fn l1<S, T>(signal: ArrayView2<T>, range: Range<usize>) -> S
where
    S: SumAccumulator<T>,
    T: Float + TotalOrder,
{
    // Total loss across all axes
    let mut total = S::zero();

    signal.columns().into_iter().for_each(|column| {
        // Take the sub slice of the 2D object
        let sub = column.slice(ndarray::s!(range.clone()));

        // Calculate the median
        let median = median(sub);

        sub.iter()
            .for_each(|signal| total += (*signal - median).abs());
    });

    total
}

/// L2 loss function.
#[inline]
fn l2<S, T>(signal: ArrayView2<T>, range: Range<usize>) -> S
where
    S: SumAccumulator<T>,
    T: Float,
{
    // Total loss across all axes
    let mut total = S::zero();

    // How many rows there are
    let rows_length = T::from(range.clone().count()).unwrap_or_else(T::zero);

    signal.columns().into_iter().for_each(|column| {
        // Take the sub slice of the 2D object
        let sub = column.slice(ndarray::s!(range.clone()));

        // Calculate variance
        let mean = sub.iter().copied().sum_with_accumulator::<S>() / rows_length;

        sub.iter()
            .for_each(|value| total += (*value - mean).powi(2));
    });

    total
}

/// Fast median calculation.
#[inline]
fn median<T>(array: ArrayView1<T>) -> T
where
    T: Float + TotalOrder,
{
    let len = array.len();

    // Check the easy cases
    match len {
        0 => return T::zero(),
        1 => return array[0],
        // Midpoint is not available for num_traits unfortunately
        2 => return (array[0] + array[1]) / T::from(2.0).unwrap_or_else(T::zero),
        _ => (),
    }

    let mut array = array.to_vec();

    // Handle the case of even and odd arrays
    if len.is_multiple_of(2) {
        // Take the two middle values
        let (_, left, rest) = array.select_nth_unstable_by(len / 2 - 1, T::total_cmp);
        let (_, right, _) = rest.select_nth_unstable_by(0, T::total_cmp);

        // Midpoint is not available for num_traits unfortunately
        (*left + *right) / T::from(2.0).unwrap_or_else(T::zero)
    } else {
        // Take the single midpoint value
        let (_, mid, _) = array.select_nth_unstable_by(len / 2, T::total_cmp);

        *mid
    }
}

#[cfg(test)]
mod tests {
    use accurate::{sum::Kahan, traits::SumAccumulator as _};

    /// Check the L1 cost function.
    #[test]
    fn l1() {
        let array = ndarray::array![[10.0], [30.0], [20.0]];
        assert_eq!(super::l1::<Kahan<f64>, _>(array.view(), 0..3).sum(), 20.0);
    }

    /// Check the L2 cost function.
    #[test]
    fn l2() {
        let array = ndarray::array![[10.0], [30.0], [20.0]];
        assert_eq!(super::l2::<Kahan<f64>, _>(array.view(), 0..3).sum(), 200.0);
    }

    /// Check the median function.
    #[test]
    fn median() {
        let array = ndarray::array![10.0, 30.0, 20.0];
        assert_eq!(super::median(array.view()), 20.0);
    }
}
