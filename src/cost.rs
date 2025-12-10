//! Cost functions.

use std::ops::Range;

use medians::Medianf64 as _;
use ndarray::ArrayView2;

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
                        let sub_vec = sub.to_vec();
                        let median = sub_vec.medf_unchecked();

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
}
