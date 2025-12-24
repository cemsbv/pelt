//! Array dimensions-specific implementation.

use std::ops::Range;

use fearless_simd::Level;
use ndarray::{ArrayView, ArrayView1, ArrayView2, Dimension, Ix1, Ix2};

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
    /// Amount of rows.
    #[doc(hidden)]
    fn len_or_nrows(array: &ArrayView<f64, Self>) -> usize;

    /// Convert to 1D if possible.
    #[doc(hidden)]
    fn try_as_1d<'a>(array: &'a ArrayView<f64, Self>) -> Option<ArrayView1<'a, f64>>;

    /// L1 cost function.
    #[doc(hidden)]
    fn l1(signal: &ArrayView<f64, Self>, range: Range<usize>) -> f64;

    /// L2 cost function.
    #[doc(hidden)]
    fn l2(simd_level: Level, signal: &ArrayView<f64, Self>, range: Range<usize>) -> f64;
}

impl OneOrTwoDimensions for Ix1 {
    #[inline]
    fn len_or_nrows(array: &ArrayView1<f64>) -> usize {
        array.len()
    }

    #[inline]
    fn l1(signal: &ArrayView1<f64>, range: Range<usize>) -> f64 {
        crate::cost::l1_1d(signal, range)
    }

    #[inline]
    fn l2(simd_level: Level, signal: &ArrayView1<f64>, range: Range<usize>) -> f64 {
        crate::cost::l2_1d(simd_level, signal, range)
    }

    #[inline]
    fn try_as_1d<'a>(_array: &'a ArrayView1<f64>) -> Option<ArrayView1<'a, f64>> {
        None
    }
}

impl OneOrTwoDimensions for Ix2 {
    #[inline]
    fn len_or_nrows(array: &ArrayView2<f64>) -> usize {
        array.nrows()
    }

    #[inline]
    fn l1(signal: &ArrayView2<f64>, range: Range<usize>) -> f64 {
        crate::cost::l1_2d(signal, range)
    }

    #[inline]
    fn l2(simd_level: Level, signal: &ArrayView2<f64>, range: Range<usize>) -> f64 {
        crate::cost::l2_2d(simd_level, signal, range)
    }

    #[inline]
    fn try_as_1d<'a>(array: &'a ArrayView<f64, Self>) -> Option<ArrayView1<'a, f64>> {
        (array.ncols() == 1).then(|| array.column(0))
    }
}
