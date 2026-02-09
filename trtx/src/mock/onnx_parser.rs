//! Mock ONNX parser implementations

use crate::error::Result;
use crate::logger::Logger;
use crate::network::NetworkDefinition;

use super::from_ffi;

/// ONNX parser (mock mode)
pub struct OnnxParser {
    inner: *mut trtx_sys::TrtxOnnxParser,
}

impl OnnxParser {
    pub fn new(network: &mut NetworkDefinition, logger: &Logger) -> Result<Self> {
        let parser_ptr = trtx_onnx_parser_create(
            network.as_mut_ptr() as *mut trtx_sys::TrtxNetworkDefinition,
            logger.as_ptr(),
        )?;
        Ok(OnnxParser { inner: parser_ptr })
    }

    pub fn parse(&self, model_bytes: &[u8]) -> Result<()> {
        parse(self.inner, model_bytes)
    }
}

impl Drop for OnnxParser {
    fn drop(&mut self) {
        destroy_parser(self.inner);
    }
}

unsafe impl Send for OnnxParser {}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

fn trtx_onnx_parser_create(
    network_ptr: *mut trtx_sys::TrtxNetworkDefinition,
    logger_ptr: *mut trtx_sys::TrtxLogger,
) -> Result<*mut trtx_sys::TrtxOnnxParser> {
    let mut parser_ptr: *mut trtx_sys::TrtxOnnxParser = std::ptr::null_mut();
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_onnx_parser_create(
            network_ptr,
            logger_ptr,
            &mut parser_ptr,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }

    Ok(parser_ptr)
}

fn parse(parser_ptr: *mut trtx_sys::TrtxOnnxParser, model_bytes: &[u8]) -> Result<()> {
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_onnx_parser_parse(
            parser_ptr,
            model_bytes.as_ptr() as *const std::ffi::c_void,
            model_bytes.len(),
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }

    Ok(())
}

fn destroy_parser(parser_ptr: *mut trtx_sys::TrtxOnnxParser) {
    if !parser_ptr.is_null() {
        unsafe {
            trtx_sys::trtx_onnx_parser_destroy(parser_ptr);
        }
    }
}
