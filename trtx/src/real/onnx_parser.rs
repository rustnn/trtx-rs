//! Real TensorRT ONNX parser implementation

use crate::error::{Error, Result};
use crate::logger::Logger;
use crate::network::NetworkDefinition;

/// ONNX parser (real mode)
pub struct OnnxParser {
    inner: *mut std::ffi::c_void,
}

impl OnnxParser {
    #[cfg(not(any(
        feature = "link_tensorrt_onnxparser",
        feature = "dlopen_tensorrt_onnxparser"
    )))]
    pub fn new(network: &mut NetworkDefinition, logger: &Logger) -> Result<Self> {
        Err(Error::TrtOnnxParserLibraryNotLoaded)
    }

    #[cfg(any(
        feature = "link_tensorrt_onnxparser",
        feature = "dlopen_tensorrt_onnxparser"
    ))]
    pub fn new(network: &mut NetworkDefinition, logger: &Logger) -> Result<Self> {
        let network_ptr = network.as_mut_ptr();
        let logger_ptr = logger.as_logger_ptr();
        let parser_ptr = {
            #[cfg(feature = "link_tensorrt_onnxparser")]
            unsafe {
                trtx_sys::create_onnx_parser(network_ptr, logger_ptr)
            }
            #[cfg(not(feature = "link_tensorrt_onnxparser"))]
            #[cfg(feature = "dlopen_tensorrt_rtx")]
            unsafe {
                use libloading::Symbol;
                use std::ffi::c_void;

                use crate::TRT_ONNXPARSER_LIB;

                let lock = TRT_ONNXPARSER_LIB
                    .read()
                    .map_err(|_| Error::LockPoisining)?;
                let create_onnx_parser: Symbol<fn(*mut c_void, *mut c_void, u32) -> *mut c_void> =
                    lock.as_ref()
                        .ok_or(Error::TrtOnnxParserLibraryNotLoaded)?
                        .get(b"createNvOnnxParser_INTERNAL")?;
                create_onnx_parser(network_ptr, logger_ptr, trtx_sys::get_tensorrt_version())
            }
        };
        if parser_ptr.is_null() {
            return Err(Error::Runtime("Failed to create ONNX parser".to_string()));
        }
        Ok(OnnxParser { inner: parser_ptr })
    }

    pub fn parse(&self, model_bytes: &[u8]) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid parser".to_string()));
        }
        let success = unsafe {
            trtx_sys::parser_parse(
                self.inner,
                model_bytes.as_ptr() as *const std::ffi::c_void,
                model_bytes.len(),
            )
        };
        if !success {
            let error_msg = unsafe {
                let num_errors = trtx_sys::parser_get_nb_errors(self.inner);
                if num_errors > 0 {
                    let err_ptr = trtx_sys::parser_get_error(self.inner, 0);
                    if !err_ptr.is_null() {
                        let desc_ptr = trtx_sys::parser_error_desc(err_ptr);
                        if !desc_ptr.is_null() {
                            std::ffi::CStr::from_ptr(desc_ptr)
                                .to_str()
                                .unwrap_or("Failed to parse ONNX model")
                                .to_string()
                        } else {
                            "Failed to parse ONNX model".to_string()
                        }
                    } else {
                        "Failed to parse ONNX model".to_string()
                    }
                } else {
                    "Failed to parse ONNX model".to_string()
                }
            };
            return Err(Error::Runtime(error_msg));
        }
        Ok(())
    }
}

impl Drop for OnnxParser {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_sys::delete_parser(self.inner);
            }
        }
    }
}

unsafe impl Send for OnnxParser {}
