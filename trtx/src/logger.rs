//! Logger interface for TensorRT-RTX

use crate::error::Result;
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;

/// Severity level for log messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum Severity {
    /// Internal error (most severe)
    InternalError = 0,
    /// Error
    Error = 1,
    /// Warning
    Warning = 2,
    /// Info
    Info = 3,
    /// Verbose (most detailed)
    Verbose = 4,
}

/// Trait for handling log messages from TensorRT
pub trait LogHandler: Send + Sync {
    /// Called when TensorRT emits a log message
    fn log(&self, severity: Severity, message: &str);
}

/// Default logger that prints to stderr
#[derive(Debug)]
pub struct StderrLogger;

impl LogHandler for StderrLogger {
    fn log(&self, severity: Severity, message: &str) {
        eprintln!("[TensorRT {:?}] {}", severity, message);
    }
}

/// Logger wrapper that interfaces with TensorRT-RTX
pub struct Logger {
    #[cfg(not(feature = "mock"))]
    bridge: *mut trtx_sys::RustLoggerBridge,
    #[cfg(not(feature = "mock"))]
    user_data: *mut std::ffi::c_void, // Keep track of user_data for cleanup
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxLogger,
    #[cfg(feature = "mock")]
    _handler: Box<dyn LogHandler>,
}

impl Logger {
    /// Create a new logger with a custom handler
    pub fn new<H: LogHandler + 'static>(handler: H) -> Result<Self> {
        let handler_box: Box<dyn LogHandler> = Box::new(handler);
        // Double-box so we have a stable pointer to pass around
        let user_data = Box::into_raw(Box::new(handler_box)) as *mut c_void;

        #[cfg(not(feature = "mock"))]
        {
            let bridge =
                unsafe { trtx_sys::create_rust_logger_bridge(Self::log_callback, user_data) };

            if bridge.is_null() {
                // Clean up user_data (double-boxed)
                unsafe {
                    let outer = Box::from_raw(user_data as *mut Box<dyn LogHandler>);
                    let _ = *outer; // Drop the inner box
                }
                return Err(crate::error::Error::Runtime(
                    "Failed to create logger bridge".to_string(),
                ));
            }

            // DON'T reconstruct - keep user_data alive for the callback
            // We'll clean it up in Drop
            Ok(Logger { bridge, user_data })
        }

        #[cfg(feature = "mock")]
        {
            let mut logger_ptr: *mut trtx_sys::TrtxLogger = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_logger_create(
                    Some(Self::log_callback_mock),
                    user_data,
                    &mut logger_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                // Clean up user_data (double-boxed)
                unsafe {
                    let outer = Box::from_raw(user_data as *mut Box<dyn LogHandler>);
                    let _ = *outer; // Drop the inner box
                }
                return Err(crate::error::Error::from_ffi(result, &error_msg));
            }

            // Reconstruct the double-boxed handler
            let outer_box = unsafe { Box::from_raw(user_data as *mut Box<dyn LogHandler>) };
            let handler_box = *outer_box; // Extract inner box

            Ok(Logger {
                inner: logger_ptr,
                _handler: handler_box,
            })
        }
    }

    /// Create a logger that prints to stderr
    pub fn stderr() -> Result<Self> {
        Self::new(StderrLogger)
    }

    /// Get the raw ILogger pointer (for internal use with autocxx)
    #[cfg(not(feature = "mock"))]
    pub(crate) fn as_logger_ptr(&self) -> *mut c_void {
        unsafe { trtx_sys::get_logger_interface(self.bridge) }
    }

    /// Get the raw pointer (for internal use in mock mode)
    #[cfg(feature = "mock")]
    pub(crate) fn as_ptr(&self) -> *mut trtx_sys::TrtxLogger {
        self.inner
    }

    /// C callback function that bridges to Rust trait (real mode)
    #[cfg(not(feature = "mock"))]
    extern "C" fn log_callback(user_data: *mut c_void, severity: i32, msg: *const c_char) {
        if user_data.is_null() || msg.is_null() {
            return;
        }

        unsafe {
            // user_data is *mut Box<dyn LogHandler> (after we pass it through the bridge)
            let handler_box = &*(user_data as *const Box<dyn LogHandler>);
            let msg_str = CStr::from_ptr(msg);

            let severity = match severity {
                0 => Severity::InternalError,
                1 => Severity::Error,
                2 => Severity::Warning,
                3 => Severity::Info,
                4 => Severity::Verbose,
                _ => Severity::Verbose, // Default fallback
            };

            if let Ok(msg) = msg_str.to_str() {
                handler_box.log(severity, msg);
            }
        }
    }

    /// C callback function that bridges to Rust trait (mock mode)
    #[cfg(feature = "mock")]
    extern "C" fn log_callback_mock(
        user_data: *mut c_void,
        severity: trtx_sys::TrtxLoggerSeverity,
        msg: *const c_char,
    ) {
        if user_data.is_null() || msg.is_null() {
            return;
        }

        unsafe {
            let handler = &*(user_data as *const Box<dyn LogHandler>);
            let msg_str = CStr::from_ptr(msg);

            let severity = match severity {
                trtx_sys::TrtxLoggerSeverity::TRTX_SEVERITY_INTERNAL_ERROR => {
                    Severity::InternalError
                }
                trtx_sys::TrtxLoggerSeverity::TRTX_SEVERITY_ERROR => Severity::Error,
                trtx_sys::TrtxLoggerSeverity::TRTX_SEVERITY_WARNING => Severity::Warning,
                trtx_sys::TrtxLoggerSeverity::TRTX_SEVERITY_INFO => Severity::Info,
                trtx_sys::TrtxLoggerSeverity::TRTX_SEVERITY_VERBOSE => Severity::Verbose,
            };

            if let Ok(msg) = msg_str.to_str() {
                handler.log(severity, msg);
            }
        }
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        #[cfg(not(feature = "mock"))]
        {
            if !self.bridge.is_null() {
                unsafe {
                    trtx_sys::destroy_rust_logger_bridge(self.bridge);
                }
            }
            // Clean up the user_data (double-boxed handler)
            if !self.user_data.is_null() {
                unsafe {
                    let outer = Box::from_raw(self.user_data as *mut Box<dyn LogHandler>);
                    let _ = *outer; // Drop the inner box
                }
            }
        }

        #[cfg(feature = "mock")]
        {
            if !self.inner.is_null() {
                unsafe {
                    trtx_sys::trtx_logger_destroy(self.inner);
                }
            }
        }
    }
}

// Logger must be Send and Sync to be used across threads
unsafe impl Send for Logger {}
unsafe impl Sync for Logger {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[allow(dead_code)]
    #[derive(Clone)]
    struct TestLogger {
        messages: Arc<Mutex<Vec<(Severity, String)>>>,
    }

    #[allow(dead_code)]
    impl TestLogger {
        fn new() -> Self {
            Self {
                messages: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn get_messages(&self) -> Vec<(Severity, String)> {
            self.messages.lock().unwrap().clone()
        }
    }

    impl LogHandler for TestLogger {
        fn log(&self, severity: Severity, message: &str) {
            self.messages
                .lock()
                .unwrap()
                .push((severity, message.to_string()));
        }
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::InternalError < Severity::Error);
        assert!(Severity::Error < Severity::Warning);
        assert!(Severity::Warning < Severity::Info);
        assert!(Severity::Info < Severity::Verbose);
    }
}
