//! Array dimensions-specific implementation.

use std::ops::Range;

use ndarray::{ArrayView, ArrayView1, ArrayView2, Dimension, Ix1, Ix2};

use crate::{
    SegmentCostFunction,
    cost::{Cost1D, Cost2D},
};

/// Don't allow other crates to implement this.
mod sealed {
    use ndarray::{Ix1, Ix2};

    /// Don't allow other crates to implement this.
    pub trait Sealed {}

    impl Sealed for Ix1 {}
    impl Sealed for Ix2 {}
}

/// Trait allowing the input array to be both 1 and two dimensional.
pub trait OneOrTwoDimensions: Dimension + sealed::Sealed {
    type PrecalculationOutput;

    /// Amount of rows.
    #[doc(hidden)]
    fn len_or_nrows(array: &ArrayView<f64, Self>) -> usize;

    /// Precalculate function.
    #[doc(hidden)]
    fn precalculate(
        cost: SegmentCostFunction,
        signal: &ArrayView<f64, Self>,
    ) -> Self::PrecalculationOutput;

    /// Calculate the loss.
    #[doc(hidden)]
    fn loss(
        cost: &Self::PrecalculationOutput,
        signal: &ArrayView<f64, Self>,
        range: Range<usize>,
    ) -> f64;

    /// Convert to 1D if possible.
    #[doc(hidden)]
    fn try_as_1d<'a>(array: &'a ArrayView<f64, Self>) -> Option<ArrayView1<'a, f64>>;
}

impl OneOrTwoDimensions for Ix1 {
    type PrecalculationOutput = Cost1D;

    #[inline]
    fn len_or_nrows(array: &ArrayView1<f64>) -> usize {
        array.len()
    }

    #[inline]
    fn precalculate(
        cost: SegmentCostFunction,
        signal: &ArrayView1<f64>,
    ) -> Self::PrecalculationOutput {
        Self::PrecalculationOutput::precalculate(cost, signal)
    }

    #[inline]
    fn loss(
        cost: &Self::PrecalculationOutput,
        signal: &ArrayView1<f64>,
        range: Range<usize>,
    ) -> f64 {
        cost.loss(signal, range)
    }

    #[inline]
    fn try_as_1d<'a>(_array: &'a ArrayView1<f64>) -> Option<ArrayView1<'a, f64>> {
        None
    }
}

impl OneOrTwoDimensions for Ix2 {
    type PrecalculationOutput = Cost2D;

    #[inline]
    fn len_or_nrows(array: &ArrayView2<f64>) -> usize {
        array.nrows()
    }

    #[inline]
    fn precalculate(
        cost: SegmentCostFunction,
        signal: &ArrayView2<f64>,
    ) -> Self::PrecalculationOutput {
        Self::PrecalculationOutput::precalculate(cost, signal)
    }

    #[inline]
    fn loss(
        cost: &Self::PrecalculationOutput,
        signal: &ArrayView2<f64>,
        range: Range<usize>,
    ) -> f64 {
        cost.loss(signal, range)
    }

    #[inline]
    fn try_as_1d<'a>(array: &'a ArrayView<f64, Self>) -> Option<ArrayView1<'a, f64>> {
        (array.ncols() == 1).then(|| array.column(0))
    }
}
