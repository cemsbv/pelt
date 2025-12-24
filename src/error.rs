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

#[cfg(feature = "rayon")]
impl Error {
    /// Convert enum to a number, used for error handling.
    pub(crate) const fn into_error_u8(self) -> u8 {
        // 0 is "ok"
        match self {
            Self::NotEnoughPoints => 1,
            Self::NoSegmentsFound => 2,
        }
    }

    /// Convert number to result, used for error handling.
    ///
    /// # Panics
    ///
    /// - If number is not a valid enum variant.
    #[allow(clippy::panic_in_result_fn)]
    pub(crate) const fn try_from_u8(error_number: u8) -> Result<(), Self> {
        match error_number {
            0 => Ok(()),
            1 => Err(Self::NotEnoughPoints),
            2 => Err(Self::NoSegmentsFound),
            _ => panic!("Unrecognized error number"),
        }
    }
}
