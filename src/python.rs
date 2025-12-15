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

    use numpy::{PyArray1, PyArrayLike2};
    use pyo3::{exceptions::PyValueError, prelude::*};

    use crate::{Kahan, Naive, Pelt, SegmentCostFunction};

    /// Calculate the changepoints.
    #[pyfunction(signature = (signal, penalty, segment_cost_function = "l1", sum_method = "kahan", jump = 10, minimum_segment_length = 2))]
    fn predict<'py>(
        py: Python<'py>,
        signal: PyArrayLike2<'py, f64>,
        penalty: f64,
        segment_cost_function: &str,
        sum_method: &str,
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

        let indices = match sum_method {
            "kahan" => setup.predict::<Kahan<_>, _>(signal.as_array(), penalty)?,
            "naive" => setup.predict::<Naive<_>, _>(signal.as_array(), penalty)?,
            // Handle unknown case
            _ => {
                return Err(PyValueError::new_err(
                    "sum_method must be 'kahan' or 'naive'",
                ));
            }
        };

        Ok(PyArray1::from_vec(py, indices))
    }
}
