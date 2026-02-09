//! Mock logger implementations

use crate::error::Result;
use crate::logger::{LogHandler, Severity};
use std::os::raw::c_char;

/// Logger (mock mode)
pub struct Logger {
    pub(crate) inner: *mut trtx_sys::TrtxLogger,
    _handler: Box<dyn LogHandler>,
}

impl Logger {
    pub fn new<H: LogHandler + 'static>(handler: H) -> Result<Self> {
        let handler_box: Box<dyn LogHandler> = Box::new(handler);
        let user_data = Box::into_raw(Box::new(handler_box)) as *mut std::ffi::c_void;

        let logger_ptr = trtx_logger_create(user_data, Some(log_callback_mock))?;

        let outer_box = unsafe { Box::from_raw(user_data as *mut Box<dyn LogHandler>) };
        let handler_box = *outer_box;

        Ok(Logger {
            inner: logger_ptr,
            _handler: handler_box,
        })
    }

    pub fn stderr() -> Result<Self> {
        Self::new(crate::logger::StderrLogger)
    }

    pub(crate) fn as_ptr(&self) -> *mut trtx_sys::TrtxLogger {
        self.inner
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        trtx_logger_destroy(self.inner);
    }
}

unsafe impl Send for Logger {}
unsafe impl Sync for Logger {}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

fn trtx_logger_create(
    user_data: *mut std::ffi::c_void,
    callback: trtx_sys::TrtxLoggerCallback,
) -> Result<*mut trtx_sys::TrtxLogger> {
    let mut logger_ptr: *mut trtx_sys::TrtxLogger = std::ptr::null_mut();
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_logger_create(
            callback,
            user_data,
            &mut logger_ptr,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(super::from_ffi(result, &error_msg));
    }

    Ok(logger_ptr)
}

fn trtx_logger_destroy(logger_ptr: *mut trtx_sys::TrtxLogger) {
    if !logger_ptr.is_null() {
        unsafe {
            trtx_sys::trtx_logger_destroy(logger_ptr);
        }
    }
}

extern "C" fn log_callback_mock(
    user_data: *mut std::ffi::c_void,
    severity: trtx_sys::TrtxLoggerSeverity,
    msg: *const c_char,
) {
    if user_data.is_null() || msg.is_null() {
        return;
    }

    unsafe {
        let handler = &*(user_data as *const Box<dyn LogHandler>);
        let msg_str = std::ffi::CStr::from_ptr(msg);

        let severity = match severity {
            trtx_sys::TrtxLoggerSeverity::TRTX_SEVERITY_INTERNAL_ERROR => Severity::InternalError,
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
