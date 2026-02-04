//! Error types for TensorRT-RTX operations

use std::ffi::NulError;
use thiserror::Error;

/// Result type for TensorRT-RTX operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when using TensorRT-RTX
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid argument provided to function
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Out of memory
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Runtime error from TensorRT
    #[error("Runtime error: {0}")]
    Runtime(String),

    /// CUDA error
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// Unknown error
    #[error("Unknown error: {0}")]
    Unknown(String),

    /// String conversion error
    #[error("String conversion error: {0}")]
    StringConversion(#[from] NulError),

    /// UTF-8 conversion error
    #[error("UTF-8 conversion error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl Error {
    /// Create error from FFI error code and message buffer (mock mode)
    #[cfg(feature = "mock")]
    pub(crate) fn from_ffi(code: i32, error_msg: &[i8]) -> Self {
        let msg = Self::parse_error_msg(error_msg);

        match code {
            code if code == trtx_sys::TRTX_ERROR_INVALID_ARGUMENT as i32 => {
                Error::InvalidArgument(msg)
            }
            code if code == trtx_sys::TRTX_ERROR_OUT_OF_MEMORY as i32 => Error::OutOfMemory(msg),
            code if code == trtx_sys::TRTX_ERROR_RUNTIME_ERROR as i32 => Error::Runtime(msg),
            code if code == trtx_sys::TRTX_ERROR_CUDA_ERROR as i32 => Error::Cuda(msg),
            _ => Error::Unknown(msg),
        }
    }

    /// Parse error message from C string buffer (mock mode)
    #[cfg(feature = "mock")]
    fn parse_error_msg(buffer: &[i8]) -> String {
        // Find null terminator
        let len = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());

        // Convert i8 to u8 safely
        let bytes: Vec<u8> = buffer[..len].iter().map(|&c| c as u8).collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::InvalidArgument("test".to_string());
        assert_eq!(err.to_string(), "Invalid argument: test");
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_parse_error_msg() {
        let msg = b"test error\0".map(|b| b as i8);
        let parsed = Error::parse_error_msg(&msg);
        assert_eq!(parsed, "test error");
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_from_ffi() {
        let msg = b"test\0".map(|b| b as i8);
        let err = Error::from_ffi(trtx_sys::TRTX_ERROR_INVALID_ARGUMENT as i32, &msg);
        match err {
            Error::InvalidArgument(s) => assert_eq!(s, "test"),
            _ => panic!("Wrong error type"),
        }
    }
}
