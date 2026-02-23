//! L1 cost functions.

use std::ops::Range;

use ndarray::{ArrayView1, ArrayView2};

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
    /// Check the L1 cost function.
    #[test]
    fn cost() {
        let array_1d = ndarray::array![10.0, 30.0, 20.0];
        assert_eq!(super::l1_1d(&array_1d.view(), 0..3), 20.0);

        let array_2d = ndarray::array![[10.0], [30.0], [20.0]];
        assert_eq!(super::l1_2d(&array_2d.view(), 0..3), 20.0);
    }

    /// Check the median function.
    #[test]
    fn median() {
        let array = ndarray::array![10.0, 30.0, 20.0];
        assert_eq!(super::median(array.view()), 20.0);
    }
}
