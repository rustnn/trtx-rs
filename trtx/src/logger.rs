//! Logger interface for TensorRT-RTX

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

pub struct LogCrateLogger;

impl LogHandler for LogCrateLogger {
    fn log(&self, severity: Severity, message: &str) {
        match severity {
            Severity::InternalError => log::error!(target: "trtx::tensorrt", "{message}"),
            Severity::Error => log::error!(target: "trtx::tensorrt", "{message}"),
            Severity::Warning => log::warn!(target: "trtx::tensorrt", "{message}"),
            Severity::Info => log::info!(target: "trtx::tensorrt", "{message}"),
            Severity::Verbose => log::debug!(target: "trtx::tensorrt", "{message}"),
        }
    }
}

/// Logger (uses Rust bridge to TensorRT)
pub struct Logger {
    bridge: *mut trtx_sys::RustLoggerBridge,
    user_data: *mut std::ffi::c_void,
}

impl Logger {
    pub fn new<H: LogHandler + 'static>(handler: H) -> crate::Result<Self> {
        let handler_box: Box<dyn LogHandler> = Box::new(handler);
        let user_data = Box::into_raw(Box::new(handler_box)) as *mut std::ffi::c_void;

        let bridge = unsafe { trtx_sys::create_rust_logger_bridge(Self::log_callback, user_data) };

        if bridge.is_null() {
            unsafe {
                let outer = Box::from_raw(user_data as *mut Box<dyn LogHandler>);
                let _ = *outer;
            }
            return Err(crate::error::Error::Runtime(
                "Failed to create logger bridge".to_string(),
            ));
        }

        Ok(Logger { bridge, user_data })
    }

    pub fn stderr() -> crate::Result<Self> {
        Self::new(StderrLogger)
    }

    pub fn log_crate() -> crate::Result<Self> {
        Self::new(LogCrateLogger)
    }

    pub(crate) fn as_logger_ptr(&self) -> *mut std::ffi::c_void {
        unsafe { trtx_sys::get_logger_interface(self.bridge) }
    }

    extern "C" fn log_callback(
        user_data: *mut std::ffi::c_void,
        severity: i32,
        msg: *const std::os::raw::c_char,
    ) {
        if user_data.is_null() || msg.is_null() {
            return;
        }
        unsafe {
            let handler_box = &*(user_data as *const Box<dyn LogHandler>);
            let msg_str = std::ffi::CStr::from_ptr(msg);
            let severity = match severity {
                0 => Severity::InternalError,
                1 => Severity::Error,
                2 => Severity::Warning,
                3 => Severity::Info,
                4 => Severity::Verbose,
                _ => Severity::Verbose,
            };
            if let Ok(msg) = msg_str.to_str() {
                handler_box.log(severity, msg);
            }
        }
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        if !self.bridge.is_null() {
            unsafe {
                trtx_sys::destroy_rust_logger_bridge(self.bridge);
            }
        }
        if !self.user_data.is_null() {
            unsafe {
                let outer = Box::from_raw(self.user_data as *mut Box<dyn LogHandler>);
                let _ = *outer;
            }
        }
    }
}

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
