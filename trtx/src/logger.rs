//! Logger interface for TensorRT-RTX

use crate::error::Result;
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;
use trtx_sys::*;

/// Severity level for log messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum Severity {
    /// Internal error (most severe)
    InternalError = TrtxLoggerSeverity_TRTX_SEVERITY_INTERNAL_ERROR,
    /// Error
    Error = TrtxLoggerSeverity_TRTX_SEVERITY_ERROR,
    /// Warning
    Warning = TrtxLoggerSeverity_TRTX_SEVERITY_WARNING,
    /// Info
    Info = TrtxLoggerSeverity_TRTX_SEVERITY_INFO,
    /// Verbose (most detailed)
    Verbose = TrtxLoggerSeverity_TRTX_SEVERITY_VERBOSE,
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
    inner: *mut TrtxLogger,
    // Keep the handler alive
    _handler: Box<dyn LogHandler>,
}

impl Logger {
    /// Create a new logger with a custom handler
    pub fn new<H: LogHandler + 'static>(handler: H) -> Result<Self> {
        let handler_box: Box<dyn LogHandler> = Box::new(handler);
        let user_data = Box::into_raw(Box::new(handler_box)) as *mut c_void;

        let mut logger_ptr: *mut TrtxLogger = std::ptr::null_mut();
        let mut error_msg = [0i8; 1024];

        let result = unsafe {
            trtx_logger_create(
                Some(Self::log_callback),
                user_data,
                &mut logger_ptr,
                error_msg.as_mut_ptr(),
                error_msg.len(),
            )
        };

        if result != TRTX_SUCCESS as i32 {
            // Clean up user_data
            unsafe {
                let _ = Box::from_raw(user_data as *mut Box<dyn LogHandler>);
            }
            return Err(crate::error::Error::from_ffi(result, &error_msg));
        }

        // Reconstruct the handler box for keeping alive
        let handler_box = unsafe { Box::from_raw(user_data as *mut Box<dyn LogHandler>) };

        Ok(Logger {
            inner: logger_ptr,
            _handler: *handler_box,
        })
    }

    /// Create a logger that prints to stderr
    pub fn stderr() -> Result<Self> {
        Self::new(StderrLogger)
    }

    /// Get the raw pointer (for internal use)
    pub(crate) fn as_ptr(&self) -> *mut TrtxLogger {
        self.inner
    }

    /// C callback function that bridges to Rust trait
    extern "C" fn log_callback(
        user_data: *mut c_void,
        severity: TrtxLoggerSeverity,
        msg: *const c_char,
    ) {
        if user_data.is_null() || msg.is_null() {
            return;
        }

        unsafe {
            let handler = &*(user_data as *const Box<dyn LogHandler>);
            let msg_str = CStr::from_ptr(msg);

            let severity = match severity {
                TrtxLoggerSeverity_TRTX_SEVERITY_INTERNAL_ERROR => Severity::InternalError,
                TrtxLoggerSeverity_TRTX_SEVERITY_ERROR => Severity::Error,
                TrtxLoggerSeverity_TRTX_SEVERITY_WARNING => Severity::Warning,
                TrtxLoggerSeverity_TRTX_SEVERITY_INFO => Severity::Info,
                TrtxLoggerSeverity_TRTX_SEVERITY_VERBOSE => Severity::Verbose,
                _ => Severity::Verbose, // Default fallback
            };

            if let Ok(msg) = msg_str.to_str() {
                handler.log(severity, msg);
            }
        }
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_logger_destroy(self.inner);
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
