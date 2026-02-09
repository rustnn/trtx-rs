//! Mock error handling - FFI error code conversion
//!
//! Delegates to Error::from_ffi in the error module to avoid circular dependency.

use crate::error::Error;

/// Create error from FFI error code and message buffer (mock mode)
pub(crate) fn from_ffi(code: i32, error_msg: &[i8]) -> Error {
    Error::from_ffi(code, error_msg)
}
