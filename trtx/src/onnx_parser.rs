//! ONNX model parser for TensorRT.
//!
//! [`OnnxParser`] wraps [`trtx_sys::nvonnxparser::IParser`] (C++ [`nvonnxparser::IParser`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvonnxparser_1_1_i_parser.html)).

use std::marker::PhantomData;

use cxx::UniquePtr;
use std::ffi::c_void;
use trtx_sys::{nvinfer1, nvonnxparser};

use crate::error::{Error, Result};
use crate::logger::Logger;
use crate::network::NetworkDefinition;

/// [`trtx_sys::nvonnxparser::IParser`] — C++ [`nvonnxparser::IParser`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvonnxparser_1_1_i_parser.html).
pub struct OnnxParser<'network> {
    inner: UniquePtr<nvonnxparser::IParser>,
    _network: PhantomData<&'network nvinfer1::INetworkDefinition>,
}

impl OnnxParser<'_> {
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
        #[cfg(not(feature = "mock"))]
        {
            let network_ptr = network.inner.as_mut_ptr();
            let logger_ptr = logger.as_logger_ptr();
            let parser_ptr = {
                #[cfg(feature = "link_tensorrt_onnxparser")]
                unsafe {
                    trtx_sys::create_onnx_parser(network_ptr, logger_ptr)
                }
                #[cfg(not(feature = "link_tensorrt_onnxparser"))]
                #[cfg(feature = "dlopen_tensorrt_onnxparser")]
                unsafe {
                    use libloading::Symbol;
                    use trtx_sys::nvinfer1::INetworkDefinition;

                    use crate::TRT_ONNXPARSER_LIB;

                    if !TRT_ONNXPARSER_LIB.read()?.is_some() {
                        crate::dynamically_load_tensorrt_onnxparser(None::<String>)?;
                    }

                    let lock = TRT_ONNXPARSER_LIB
                        .read()
                        .map_err(|_| Error::LockPoisining)?;
                    let create_onnx_parser: Symbol<
                        fn(*mut INetworkDefinition, *mut c_void, u32) -> *mut c_void,
                    > = lock
                        .as_ref()
                        .ok_or(Error::TrtOnnxParserLibraryNotLoaded)?
                        .get(b"createNvOnnxParser_INTERNAL")?;
                    create_onnx_parser(network_ptr, logger_ptr, trtx_sys::get_tensorrt_version())
                }
            } as *mut nvonnxparser::IParser;
            if parser_ptr.is_null() {
                return Err(Error::Runtime("Failed to create ONNX parser".to_string()));
            }
            Ok(OnnxParser {
                inner: unsafe { UniquePtr::from_raw(parser_ptr) },
                _network: Default::default(),
            })
        }
        #[cfg(feature = "mock")]
        Ok(OnnxParser {
            inner: UniquePtr::null(),
            _network: Default::default(),
        })
    }

    pub fn parse(&mut self, model_bytes: &[u8]) -> Result<()> {
        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid parser".to_string()));
            }
            let parser_ptr = self.inner.as_mut_ptr() as *mut c_void;
            let success = unsafe {
                trtx_sys::parser_parse(
                    parser_ptr,
                    model_bytes.as_ptr() as *const std::ffi::c_void,
                    model_bytes.len(),
                )
            };
            if !success {
                let error_msg = unsafe {
                    let num_errors = trtx_sys::parser_get_nb_errors(parser_ptr);
                    if num_errors > 0 {
                        let err_ptr = trtx_sys::parser_get_error(parser_ptr, 0);
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
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::network_flags;
    use crate::{Builder, Logger};

    #[test]
    fn test_onnx_parser_creation() {
        let logger = Logger::stderr().unwrap();
        let mut builder = Builder::new(&logger).unwrap();
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
        let mut builder = Builder::new(&logger).unwrap();
        let mut network = builder
            .create_network(network_flags::EXPLICIT_BATCH)
            .unwrap();
        let mut parser = OnnxParser::new(&mut network, &logger).unwrap();
        let result = parser.parse(&model_bytes);
        assert!(
            result.is_ok(),
            "Failed to parse ONNX model: {:?}",
            result.err()
        );
    }
}
