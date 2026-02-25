//! Cost functions.

pub(crate) mod l1;
pub(crate) mod l2;
mod tree;

use std::ops::Range;

use l2::{L2Cost1D, L2Cost2D};
use ndarray::{ArrayView1, ArrayView2};

use crate::cost::l1::{L1Cost1D, L1Cost2D};

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

/// Precalculation state for each segment cost function.
#[doc(hidden)]
pub enum Cost1D {
    /// L1.
    L1(L1Cost1D),
    /// L2.
    L2(L2Cost1D),
}

impl Cost1D {
    /// Construct from signal and cost function.
    #[inline]
    pub(crate) fn precalculate(cost: SegmentCostFunction, signal: &ArrayView1<f64>) -> Self {
        match cost {
            SegmentCostFunction::L1 => Self::L1(L1Cost1D::precalculate(signal)),
            SegmentCostFunction::L2 => Self::L2(L2Cost1D::precalculate(signal)),
        }
    }

    /// Calculate the loss.
    #[inline]
    pub(crate) fn loss(&self, signal: &ArrayView1<f64>, range: Range<usize>) -> f64 {
        match self {
            Self::L1(cost) => cost.loss(signal, range),
            Self::L2(cost) => cost.loss(range),
        }
    }
}

/// Precalculation state for each segment cost function.
#[doc(hidden)]
pub enum Cost2D {
    /// L1.
    L1(L1Cost2D),
    /// L2.
    L2(L2Cost2D),
}

impl Cost2D {
    /// Construct from signal and cost function.
    #[inline]
    pub(crate) fn precalculate(cost: SegmentCostFunction, signal: &ArrayView2<f64>) -> Self {
        match cost {
            SegmentCostFunction::L1 => Self::L1(L1Cost2D::precalculate(signal)),
            SegmentCostFunction::L2 => Self::L2(L2Cost2D::precalculate(signal)),
        }
    }

    /// Calculate the loss.
    #[inline]
    pub(crate) fn loss(&self, signal: &ArrayView2<f64>, range: Range<usize>) -> f64 {
        match self {
            Self::L1(cost) => cost.loss(signal, range),
            Self::L2(cost) => cost.loss(range),
        }
    }
}
