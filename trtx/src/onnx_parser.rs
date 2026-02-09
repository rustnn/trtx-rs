//! ONNX model parser for TensorRT
//!
//! Delegates to real/ or mock/ based on feature flag.

#[cfg(not(feature = "mock"))]
pub use crate::real::onnx_parser::OnnxParser;
#[cfg(feature = "mock")]
pub use crate::mock::onnx_parser::OnnxParser;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::network_flags;
    use crate::{Builder, Logger};

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
    #[ignore] // Requires GPU and TensorRT runtime
    fn test_onnx_parser_with_real_model() {
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
        assert!(result.is_ok(), "Failed to parse ONNX model: {:?}", result.err());
    }
}
