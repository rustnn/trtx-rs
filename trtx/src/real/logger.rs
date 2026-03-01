//! Real TensorRT logger implementation

use crate::error::Result;
use crate::logger::{LogHandler, Severity};
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;

/// Logger (real mode - uses Rust bridge)
pub struct Logger {
    bridge: *mut trtx_sys::RustLoggerBridge,
    user_data: *mut std::ffi::c_void,
}

impl Logger {
    pub fn new<H: LogHandler + 'static>(handler: H) -> Result<Self> {
        let handler_box: Box<dyn LogHandler> = Box::new(handler);
        let user_data = Box::into_raw(Box::new(handler_box)) as *mut c_void;

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

    pub fn stderr() -> Result<Self> {
        Self::new(crate::logger::StderrLogger)
    }

    pub fn log_crate() -> Result<Self> {
        Self::new(crate::logger::LogCrateLogger)
    }

    pub(crate) fn as_logger_ptr(&self) -> *mut c_void {
        unsafe { trtx_sys::get_logger_interface(self.bridge) }
    }

    extern "C" fn log_callback(user_data: *mut c_void, severity: i32, msg: *const c_char) {
        if user_data.is_null() || msg.is_null() {
            return;
        }
        unsafe {
            let handler_box = &*(user_data as *const Box<dyn LogHandler>);
            let msg_str = CStr::from_ptr(msg);
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
