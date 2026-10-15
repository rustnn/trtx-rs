//! ONNX model parser for TensorRT.
//!
//! [`OnnxParser`] wraps [`trtx_sys::nvonnxparser::IParser`] (C++ [`nvonnxparser::IParser`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvonnxparser_1_1_i_parser.html)).

use std::path::PathBuf;

use cxx::UniquePtr;
use std::ffi::{c_void, CString};
use trtx_sys::nvonnxparser;

use crate::error::{Error, Result};
use crate::logger::Logger;
use crate::network::NetworkDefinition;

/// [`trtx_sys::nvonnxparser::IParser`] — C++ [`nvonnxparser::IParser`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvonnxparser_1_1_i_parser.html).
pub struct OnnxParser<'network> {
    inner: UniquePtr<nvonnxparser::IParser>,
    #[cfg(feature = "v_1_7")]
    refit_observer: Option<RefitterObserver>,
    network: NetworkDefinition<'network>,
}

impl<'parser> OnnxParser<'parser> {
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
    /// Creates a new Parser. This consumes NetworkDefinition to ensure the parser is not dropped
    /// before the network build is finished (as the parser holds the weights).
    ///
    /// Use [OnnxParser::network], [OnnxParser::network_mut] to get a borrow to the network
    pub fn new(network: NetworkDefinition<'parser>, logger: &Logger) -> Result<Self> {
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
                    create_onnx_parser(
                        network_ptr,
                        logger_ptr,
                        trtx_sys::get_nvonnxparser_version(),
                    )
                }
            } as *mut nvonnxparser::IParser;
            if parser_ptr.is_null() {
                return Err(Error::Runtime("Failed to create ONNX parser".to_string()));
            }
            Ok(OnnxParser {
                inner: unsafe { UniquePtr::from_raw(parser_ptr) },
                #[cfg(feature = "v_1_7")]
                refit_observer: None,
                network,
            })
        }
        #[cfg(feature = "mock")]
        Ok(OnnxParser {
            inner: UniquePtr::null(),
            #[cfg(all(feature = "v_1_7", not(feature = "enterprise")))]
            refit_observer: None,
            network,
        })
    }

    /// Attaches an observer invoked once per named weight during parsing.
    /// Records are owned copies and may be retained after the callback. They form a candidate
    /// superset of engine refittable weights and are valid only when parsing succeeds.
    /// A panicking callback aborts the process, since unwinding through C++ is unsupported.
    #[cfg(feature = "v_1_7")]
    pub fn set_refit_observer(
        &mut self,
        observer: impl FnMut(RefitRecord) + 'static,
    ) -> Result<()> {
        let observer = RefitterObserver::new(Box::new(observer))?;
        #[cfg(not(feature = "mock_runtime"))]
        unsafe {
            trtx_sys::trtx_parser_set_refit_observer(self.inner.as_mut_ptr(), observer.cpp);
        }
        self.refit_observer = Some(observer);
        Ok(())
    }

    pub fn parse_from_file(&mut self, path: &str, verbosity: i32) -> Result<()> {
        let cpath = CString::new(path)?;
        unsafe {
            if self
                .inner
                .pin_mut()
                .parseFromFile(cpath.as_ptr(), verbosity.into())
            {
                Ok(())
            } else {
                Err(Error::FailedToParseOnnx(PathBuf::from(path)))
            }
        }
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

    pub fn network(&'parser self) -> &'parser NetworkDefinition<'parser> {
        &self.network
    }

    pub fn network_mut(&'parser mut self) -> &'parser mut NetworkDefinition<'parser> {
        &mut self.network
    }
}
/// Owned description of a named weight emitted while parsing an ONNX model.
#[cfg(feature = "v_1_7")]
#[derive(Debug, Clone)]
pub struct RefitRecord {
    pub trt_name: String,
    pub kind: RefitTransformKind,
    pub onnx_dtype: i32,
    pub trt_dtype: crate::DataType,
    pub count: i64,
    pub source_onnx_names: Vec<String>,
    /// Used only for batch normalization folding.
    pub epsilon: f32,
    /// Build-time bytes for Constant and ConstantOfShape nodes.
    pub fixed_data: Vec<u8>,
}

/// Transformation from ONNX source data to TensorRT refit data.
#[cfg(feature = "v_1_7")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefitTransformKind {
    Identity,
    DoubleToFloat,
    BatchNormFoldScale,
    BatchNormFoldBias,
    ConstantNode,
    ConstantOfShape,
    Unknown(i32),
}

#[cfg(feature = "v_1_7")]
struct RefitterObserver {
    cpp: *mut c_void,
    // Boxing the box gives the callback a stable, thin pointer.
    _callback: Box<Box<dyn FnMut(RefitRecord)>>,
}

#[cfg(feature = "v_1_7")]
impl RefitterObserver {
    fn new(callback: Box<dyn FnMut(RefitRecord)>) -> Result<Self> {
        let mut callback = Box::new(callback);
        let cpp = unsafe {
            trtx_sys::trtx_create_refitter_observer(
                (&mut *callback as *mut Box<dyn FnMut(RefitRecord)>).cast(),
                refit_observer_callback,
            )
        };
        if cpp.is_null() {
            return Err(Error::Runtime(
                "Failed to create refit observer".to_string(),
            ));
        }
        Ok(Self {
            cpp,
            _callback: callback,
        })
    }
}

#[cfg(feature = "v_1_7")]
impl Drop for RefitterObserver {
    fn drop(&mut self) {
        unsafe { trtx_sys::trtx_destroy_refitter_observer(self.cpp) }
    }
}

#[cfg(feature = "v_1_7")]
unsafe extern "C" fn refit_observer_callback(
    user_data: *mut c_void,
    record: *const trtx_sys::RefitRecord,
) {
    // Never unwind across the C++ noexcept callback boundary.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let record = &*record;
        let kind = match record.kind {
            0 => RefitTransformKind::Identity,
            1 => RefitTransformKind::DoubleToFloat,
            2 => RefitTransformKind::BatchNormFoldScale,
            3 => RefitTransformKind::BatchNormFoldBias,
            4 => RefitTransformKind::ConstantNode,
            5 => RefitTransformKind::ConstantOfShape,
            other => RefitTransformKind::Unknown(other),
        };
        let source_onnx_names = if record.nb_sources > 0 {
            std::slice::from_raw_parts(record.source_onnx_names, record.nb_sources as usize)
                .iter()
                .map(|name| {
                    std::ffi::CStr::from_ptr(*name)
                        .to_string_lossy()
                        .into_owned()
                })
                .collect()
        } else {
            Vec::new()
        };
        let fixed_data = if record.fixed_data_size > 0 {
            std::slice::from_raw_parts(record.fixed_data.cast::<u8>(), record.fixed_data_size)
                .to_vec()
        } else {
            Vec::new()
        };
        let callback = &mut *user_data.cast::<Box<dyn FnMut(RefitRecord)>>();
        callback(RefitRecord {
            trt_name: std::ffi::CStr::from_ptr(record.trt_name)
                .to_string_lossy()
                .into_owned(),
            kind,
            onnx_dtype: record.onnx_dtype,
            trt_dtype: std::mem::transmute::<i32, crate::DataType>(record.trt_dtype),
            count: record.count,
            source_onnx_names,
            epsilon: if matches!(
                kind,
                RefitTransformKind::BatchNormFoldScale | RefitTransformKind::BatchNormFoldBias
            ) {
                record.epsilon
            } else {
                0.0
            },
            fixed_data,
        });
    }));
    if result.is_err() {
        std::process::abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Builder, Logger};

    #[test]
    fn test_onnx_parser_creation() {
        let logger = Logger::stderr().unwrap();
        let mut builder = Builder::new(&logger).unwrap();
        let network = builder.create_network(0).unwrap();
        let parser = OnnxParser::new(network, &logger);
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
        let network = builder.create_network(0).unwrap();
        let mut parser = OnnxParser::new(network, &logger).unwrap();
        let result = parser.parse(&model_bytes);
        assert!(
            result.is_ok(),
            "Failed to parse ONNX model: {:?}",
            result.err()
        );
    }
}
