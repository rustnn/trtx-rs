//! ONNX model parser for TensorRT

use crate::builder::NetworkDefinition;
use crate::error::{Error, Result};
use crate::logger::Logger;
use trtx_sys::*;

/// ONNX model parser
pub struct OnnxParser {
    inner: *mut TrtxOnnxParser,
}

impl OnnxParser {
    /// Create a new ONNX parser for the given network
    pub fn new(network: &NetworkDefinition, logger: &Logger) -> Result<Self> {
        let mut parser_ptr: *mut TrtxOnnxParser = std::ptr::null_mut();
        let mut error_msg = [0i8; 1024];

        let result = unsafe {
            trtx_onnx_parser_create(
                network.as_ptr(),
                logger.as_ptr(),
                &mut parser_ptr,
                error_msg.as_mut_ptr(),
                error_msg.len(),
            )
        };

        if result != TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        Ok(OnnxParser { inner: parser_ptr })
    }

    /// Parse an ONNX model from bytes
    pub fn parse(&self, model_bytes: &[u8]) -> Result<()> {
        let mut error_msg = [0i8; 1024];

        let result = unsafe {
            trtx_onnx_parser_parse(
                self.inner,
                model_bytes.as_ptr() as *const std::ffi::c_void,
                model_bytes.len(),
                error_msg.as_mut_ptr(),
                error_msg.len(),
            )
        };

        if result != TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        Ok(())
    }
}

impl Drop for OnnxParser {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_onnx_parser_destroy(self.inner);
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
        let network = builder
            .create_network(network_flags::EXPLICIT_BATCH)
            .unwrap();

        let parser = OnnxParser::new(&network, &logger);
        assert!(parser.is_ok());
    }

    #[test]
    #[ignore] // Requires GPU and TensorRT runtime - run with: cargo test --ignored test_onnx_parser_with_real_model
    fn test_onnx_parser_with_real_model() {
        // Load the test ONNX model (super-resolution-10.onnx from ONNX model zoo)
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/super-resolution-10.onnx");
        let model_bytes = std::fs::read(model_path).expect("Failed to read test ONNX model");

        let logger = Logger::stderr().unwrap();
        let builder = Builder::new(&logger).unwrap();
        let network = builder
            .create_network(network_flags::EXPLICIT_BATCH)
            .unwrap();

        let parser = OnnxParser::new(&network, &logger).unwrap();
        let result = parser.parse(&model_bytes);

        // Parse should succeed with a valid ONNX model
        assert!(result.is_ok(), "Failed to parse ONNX model: {:?}", result.err());
    }
}
