//! Error types.

/// Errors that can occur during calculation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Calculated segment is too short.
    #[error("calculated segment of loss function is too short")]
    NotEnoughPoints,
    /// No segments got calculated.
    #[error("calculation didn't return any segments")]
    NoSegmentsFound,
}
