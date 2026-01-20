//! ONNX model parser for TensorRT

use crate::error::{Error, Result};
use crate::logger::Logger;
use crate::network::NetworkDefinition;

/// ONNX model parser
pub struct OnnxParser {
    #[cfg(not(feature = "mock"))]
    inner: *mut std::ffi::c_void,
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxOnnxParser,
}

impl OnnxParser {
    /// Create a new ONNX parser for the given network
    pub fn new(network: &mut NetworkDefinition, logger: &Logger) -> Result<Self> {
        #[cfg(feature = "mock")]
        {
            let mut parser_ptr: *mut trtx_sys::TrtxOnnxParser = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_onnx_parser_create(
                    network.as_ptr(),
                    logger.as_ptr(),
                    &mut parser_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(OnnxParser { inner: parser_ptr })
        }

        #[cfg(not(feature = "mock"))]
        {
            let network_ptr = network.as_mut_ptr();
            let logger_ptr = logger.as_logger_ptr();

            let parser_ptr = unsafe { trtx_sys::create_onnx_parser(network_ptr, logger_ptr) };

            if parser_ptr.is_null() {
                return Err(Error::Runtime("Failed to create ONNX parser".to_string()));
            }

            Ok(OnnxParser { inner: parser_ptr })
        }
    }

    /// Parse an ONNX model from bytes
    pub fn parse(&self, model_bytes: &[u8]) -> Result<()> {
        #[cfg(feature = "mock")]
        {
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_onnx_parser_parse(
                    self.inner,
                    model_bytes.as_ptr() as *const std::ffi::c_void,
                    model_bytes.len(),
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(())
        }

        #[cfg(not(feature = "mock"))]
        {
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
                // Try to get error details from parser
                let error_msg = unsafe {
                    let num_errors = trtx_sys::parser_get_nb_errors(self.inner);
                    if num_errors > 0 {
                        let err_ptr = trtx_sys::parser_get_error(self.inner, 0);
                        if !err_ptr.is_null() {
                            let desc_ptr = trtx_sys::parser_error_desc(err_ptr);
                            if !desc_ptr.is_null() {
                                std::ffi::CStr::from_ptr(desc_ptr)
                                    .to_string_lossy()
                                    .into_owned()
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
}

impl Drop for OnnxParser {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            #[cfg(feature = "mock")]
            unsafe {
                trtx_sys::trtx_onnx_parser_destroy(self.inner);
            }
            #[cfg(not(feature = "mock"))]
            unsafe {
                trtx_sys::delete_parser(self.inner);
            }
        }
    }
}

unsafe impl Send for OnnxParser {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::network_flags;
    use crate::Builder;
    use crate::Logger;

    #[test]
    #[ignore] // Requires TensorRT runtime initialization (can hang in test context)
    fn test_onnx_parser_creation() {
        let logger = Logger::stderr().unwrap();
        let builder = Builder::new(&logger).unwrap();
        let mut network = builder
            .create_network(network_flags::EXPLICIT_BATCH)
            .unwrap();

        let parser = OnnxParser::new(&mut network, &logger);
        assert!(parser.is_ok());
    }

    #[test]
    #[ignore] // Requires GPU and TensorRT runtime - run with: cargo test --ignored test_onnx_parser_with_real_model
    fn test_onnx_parser_with_real_model() {
        // Load the test ONNX model (super-resolution-10.onnx from ONNX model zoo)
        let model_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/super-resolution-10.onnx"
        );
        let model_bytes = std::fs::read(model_path).expect("Failed to read test ONNX model");

        let logger = Logger::stderr().unwrap();
        let builder = Builder::new(&logger).unwrap();
        let mut network = builder
            .create_network(network_flags::EXPLICIT_BATCH)
            .unwrap();

        let parser = OnnxParser::new(&mut network, &logger).unwrap();
        let result = parser.parse(&model_bytes);

        // Parse should succeed with a valid ONNX model
        assert!(
            result.is_ok(),
            "Failed to parse ONNX model: {:?}",
            result.err()
        );
    }
}
