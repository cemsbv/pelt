//! Cost functions.

use std::ops::Range;

use ndarray::{ArrayView1, ArrayView2};

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
    pub(crate) fn loss(self, signal: ArrayView2<f64>, range: Range<usize>) -> f64 {
        match self {
            Self::L1 => {
                // Total loss across all axes
                signal
                    .columns()
                    .into_iter()
                    .map(|column| {
                        // Take the sub slice of the 2D object
                        let sub = column.slice(ndarray::s!(range.clone()));

                        // Calculate the median
                        let median = median(sub);

                        sub.mapv(|signal| (signal - median).abs()).sum()
                    })
                    .sum()
            }
            Self::L2 => {
                // Total loss across all axes
                signal
                    .columns()
                    .into_iter()
                    .map(|column| {
                        // Take the sub slice of the 2D object
                        let sub = column.slice(ndarray::s!(range.clone()));

                        // Calculate variance
                        let mean = sub.sum() / sub.len() as f64;
                        sub.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
                    })
                    .sum()
            }
        }
    }
}

/// Fast median calculation.
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
    use super::*;

    /// Check the L1 cost function.
    #[test]
    fn l1() {
        let array = ndarray::array![[10.0], [30.0], [20.0]];
        assert_eq!(SegmentCostFunction::L1.loss(array.view(), 0..3), 20.0);
    }

    /// Check the L2 cost function.
    #[test]
    fn l2() {
        let array = ndarray::array![[10.0], [30.0], [20.0]];
        assert_eq!(SegmentCostFunction::L2.loss(array.view(), 0..3), 200.0);
    }

    /// Check the median function.
    #[test]
    fn median() {
        assert_eq!(
            super::median(ndarray::array![10.0, 30.0, 20.0].view()),
            20.0
        );
    }
}
