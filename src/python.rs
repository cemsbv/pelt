//! Python bindings.

use pyo3::{PyErr, exceptions::PyRuntimeError};

use crate::Error;

/// Convert Rust to Python error.
impl From<Error> for PyErr {
    fn from(err: Error) -> Self {
        PyRuntimeError::new_err(err.to_string())
    }
}

#[pyo3::pymodule]
mod pelt {
    use std::num::NonZero;

    use numpy::{Ix1, Ix2, PyArray1, PyArrayLikeDyn};
    use pyo3::{exceptions::PyValueError, prelude::*};

    use crate::{Pelt, SegmentCostFunction};

    /// Calculate the changepoints.
    ///
    /// Arguments
    /// ---------
    /// signal : :py:class:`numpy.typing.NDArray[numpy.float64] <numpy.typing.NDArray>`
    ///     1D or 2D input signal array. Can only contain numbers. ``None`` values are not accepted.
    /// penalty : float
    ///     Penalty value for each changepoint added. Larger values result in fewer
    ///     changepoints detected.
    /// segment_cost_function : str, optional
    ///     Determines how the cost of each potential segment is calculated.
    ///     Must be one of:
    ///     
    ///     * ``"l1"`` - L1 cost function (least absolute deviation)
    ///     * ``"l2"`` - L2 cost function (least squared deviation)
    ///     
    ///     Defaults to ``"l1"``.
    /// jump : int, optional
    ///     Step size between candidate changepoint positions. Must be > 0.
    ///     If ``jump = 1``, a check is done every possible prior change point, guaranteeing an exact solution, finding the true minimum of the objective function.
    ///     If ``jump > 1``, previous change points are considered at intervals of ``jump``. This speeds up the computation, but the solution becomes approximate.
    ///     Defaults to 10.
    /// minimum_segment_length : int, optional
    ///     Minimum number of allowable number of data points within a segment.
    ///     Must be positive. Defaults to 2.
    ///
    /// Returns
    /// -------
    /// :py:class:`numpy.typing.NDArray[numpy.uint64] <numpy.typing.NDArray>`
    ///     1D array of zero-based indices where changes in the signal were detected.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the array has invalid dimensions or if any of the parameters are
    ///     outside their valid ranges.
    ///
    /// Examples
    /// --------
    /// >>> from pelt import predict
    /// >>> changepoints = predict(signal, penalty=20.0, segment_cost_function="l1", jump=10, minimum_segment_length=2)
    /// >>> print(changepoints)
    ///
    #[pyfunction(signature = (signal, penalty, segment_cost_function = "l1", jump = 10, minimum_segment_length = 2))]
    fn predict<'py>(
        py: Python<'py>,
        signal: PyArrayLikeDyn<'py, f64>,
        penalty: f64,
        segment_cost_function: &str,
        jump: usize,
        minimum_segment_length: usize,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        // Map input parameter to enum
        let segment_cost_function = match segment_cost_function {
            "l1" => SegmentCostFunction::L1,
            "l2" => SegmentCostFunction::L2,
            // Handle unknown case
            _ => {
                return Err(PyValueError::new_err(
                    "segment_cost_function must be 'l1' or 'l2'",
                ));
            }
        };

        // Convert types
        let jump = NonZero::new(jump).ok_or_else(|| PyValueError::new_err("jump must be > 0"))?;
        let minimum_segment_length = NonZero::new(minimum_segment_length)
            .ok_or_else(|| PyValueError::new_err("minimum_segment_length must be > 0"))?;

        // Do calculation
        let setup = Pelt::new()
            .with_segment_cost_function(segment_cost_function)
            .with_jump(jump)
            .with_minimum_segment_length(minimum_segment_length);

        // Try to coerce the input into a dimension we can use
        let signal = signal.as_array();
        let indices = match signal.ndim() {
            1 => setup.predict(
                signal
                    .into_dimensionality::<Ix1>()
                    .map_err(|_| PyValueError::new_err("dimension mismatch"))?,
                penalty,
            )?,
            2 => setup.predict(
                signal
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| PyValueError::new_err("dimension mismatch"))?,
                penalty,
            )?,
            _ => {
                return Err(PyValueError::new_err(
                    "signal array dimensions must be 1 or 2",
                ));
            }
        };

        Ok(PyArray1::from_vec(py, indices))
    }
}
