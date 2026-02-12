//! Logger interface for TensorRT-RTX
//!
//! Delegates to real/ or mock/ based on feature flag.

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

#[derive(Debug)]
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

#[cfg(feature = "mock")]
pub use crate::mock::logger::Logger;
#[cfg(not(feature = "mock"))]
pub use crate::real::logger::Logger;

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
