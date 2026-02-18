//! Error types for TensorRT-RTX operations

use std::ffi::NulError;
use thiserror::Error;

/// Result type for TensorRT-RTX operations
pub type Result<T> = std::result::Result<T, Error>;

/// Proxy for [`trtx_sys::nvinfer1::LayerType`] with [`Debug`], [`Eq`], [`PartialEq`].
/// Use this in public APIs (e.g. [`Error::LayerCreationFailed`]) so errors are well-typed
/// without depending on the FFI enum's trait impls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LayerTypeKind {
    Convolution = 0,
    Cast = 1,
    Activation = 2,
    Pooling = 3,
    Lrn = 4,
    Scale = 5,
    Softmax = 6,
    Deconvolution = 7,
    Concatenation = 8,
    Elementwise = 9,
    Plugin = 10,
    Unary = 11,
    Padding = 12,
    Shuffle = 13,
    Reduce = 14,
    TopK = 15,
    Gather = 16,
    MatrixMultiply = 17,
    RaggedSoftmax = 18,
    Constant = 19,
    Indentity = 20,
    PluginV2 = 21,
    Slice = 22,
    Shape = 23,
    ParametricRelu = 24,
    Resize = 25,
    TripLimit = 26,
    Recurrence = 27,
    Iterator = 28,
    LoopOutput = 29,
    Select = 30,
    Fill = 31,
    Quantize = 32,
    Dequantize = 33,
    Condition = 34,
    ConditionalInput = 35,
    ConditionalOutput = 36,
    Scatter = 37,
    Einsum = 38,
    Assertion = 39,
    OneHot = 40,
    NonZero = 41,
    GridSample = 42,
    Nms = 43,
    ReverseSequence = 44,
    Normalization = 45,
    PluginV3 = 46,
    Squeeze = 47,
    Unsqueeze = 48,
    Cumulative = 49,
    DynamicQunatize = 50,
    AttentionInput = 51,
    AttentionOutput = 52,
    RotaryEmbedding = 53,
    KvCacheUpdate = 54,
}

#[cfg(not(feature = "mock"))]
impl From<trtx_sys::nvinfer1::LayerType> for LayerTypeKind {
    fn from(t: trtx_sys::nvinfer1::LayerType) -> Self {
        use trtx_sys::nvinfer1::LayerType as T;
        match t {
            T::kCONVOLUTION => Self::Convolution,
            T::kCAST => Self::Cast,
            T::kACTIVATION => Self::Activation,
            T::kPOOLING => Self::Pooling,
            T::kLRN => Self::Lrn,
            T::kSCALE => Self::Scale,
            T::kSOFTMAX => Self::Softmax,
            T::kDECONVOLUTION => Self::Deconvolution,
            T::kCONCATENATION => Self::Concatenation,
            T::kELEMENTWISE => Self::Elementwise,
            T::kPLUGIN => Self::Plugin,
            T::kUNARY => Self::Unary,
            T::kPADDING => Self::Padding,
            T::kSHUFFLE => Self::Shuffle,
            T::kREDUCE => Self::Reduce,
            T::kTOPK => Self::TopK,
            T::kGATHER => Self::Gather,
            T::kMATRIX_MULTIPLY => Self::MatrixMultiply,
            T::kRAGGED_SOFTMAX => Self::RaggedSoftmax,
            T::kCONSTANT => Self::Constant,
            T::kIDENTITY => Self::Indentity,
            T::kPLUGIN_V2 => Self::PluginV2,
            T::kSLICE => Self::Slice,
            T::kSHAPE => Self::Shape,
            T::kPARAMETRIC_RELU => Self::ParametricRelu,
            T::kRESIZE => Self::Resize,
            T::kTRIP_LIMIT => Self::TripLimit,
            T::kRECURRENCE => Self::Recurrence,
            T::kITERATOR => Self::Iterator,
            T::kLOOP_OUTPUT => Self::LoopOutput,
            T::kSELECT => Self::Select,
            T::kFILL => Self::Fill,
            T::kQUANTIZE => Self::Quantize,
            T::kDEQUANTIZE => Self::Dequantize,
            T::kCONDITION => Self::Condition,
            T::kCONDITIONAL_INPUT => Self::ConditionalInput,
            T::kCONDITIONAL_OUTPUT => Self::ConditionalOutput,
            T::kSCATTER => Self::Scatter,
            T::kEINSUM => Self::Einsum,
            T::kASSERTION => Self::Assertion,
            T::kONE_HOT => Self::OneHot,
            T::kNON_ZERO => Self::NonZero,
            T::kGRID_SAMPLE => Self::GridSample,
            T::kNMS => Self::Nms,
            T::kREVERSE_SEQUENCE => Self::ReverseSequence,
            T::kNORMALIZATION => Self::Normalization,
            T::kPLUGIN_V3 => Self::PluginV3,
            T::kSQUEEZE => Self::Squeeze,
            T::kUNSQUEEZE => Self::Unsqueeze,
            T::kCUMULATIVE => Self::Cumulative,
            T::kDYNAMIC_QUANTIZE => Self::DynamicQunatize,
            T::kATTENTION_INPUT => Self::AttentionInput,
            T::kATTENTION_OUTPUT => Self::AttentionOutput,
            T::kROTARY_EMBEDDING => Self::RotaryEmbedding,
            T::kKVCACHE_UPDATE => Self::KvCacheUpdate,
        }
    }
}

/// Errors that can occur when using TensorRT-RTX
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid argument provided to function
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Out of memory
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Runtime error from TensorRT
    #[error("Runtime error: {0}")]
    Runtime(String),

    /// CUDA error
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// Unknown error
    #[error("Unknown error: {0}")]
    Unknown(String),

    /// String conversion error
    #[error("String conversion error: {0}")]
    StringConversion(#[from] NulError),

    /// UTF-8 conversion error
    #[error("UTF-8 conversion error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TensorRT library not loaded. A successful call to trtx::dynamically_load_tensorrt is required to load the TensorRT library")]
    TrtRtxLibraryNotLoaded,

    #[error("TensorRT onnxparser library not loaded. A successful call to trtx::dynamically_load_tensorrt_onnxparser is required to load the TensorRT ONNX parser library")]
    TrtOnnxParserLibraryNotLoaded,

    #[cfg(any(
        feature = "dlopen_tensorrt_rtx",
        feature = "dlopen_tensorrt_onnxparser"
    ))]
    #[error("Dynamic loading error: {0}")]
    Libloading(#[from] libloading::Error),

    #[error("Would unwrap a poisened lock")]
    LockPoisining,

    #[cfg(not(feature = "mock"))]
    #[error("Failed to create layer: {0:?}")]
    LayerCreationFailed(LayerTypeKind),
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Error::LockPoisining
    }
}

impl Error {
    /// Create error from FFI error code and message buffer (mock mode)
    #[cfg(feature = "mock")]
    pub(crate) fn from_ffi(code: i32, error_msg: &[i8]) -> Self {
        let msg = Self::parse_error_msg(error_msg);

        match code {
            code if code == trtx_sys::TRTX_ERROR_INVALID_ARGUMENT as i32 => {
                Error::InvalidArgument(msg)
            }
            code if code == trtx_sys::TRTX_ERROR_OUT_OF_MEMORY as i32 => Error::OutOfMemory(msg),
            code if code == trtx_sys::TRTX_ERROR_RUNTIME_ERROR as i32 => Error::Runtime(msg),
            code if code == trtx_sys::TRTX_ERROR_CUDA_ERROR as i32 => Error::Cuda(msg),
            _ => Error::Unknown(msg),
        }
    }

    /// Parse error message from C string buffer (mock mode)
    #[cfg(feature = "mock")]
    fn parse_error_msg(buffer: &[i8]) -> String {
        // Find null terminator
        let len = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());

        // Convert i8 to u8 safely
        let bytes: Vec<u8> = buffer[..len].iter().map(|&c| c as u8).collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::InvalidArgument("test".to_string());
        assert_eq!(err.to_string(), "Invalid argument: test");
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_parse_error_msg() {
        let msg = b"test error\0".map(|b| b as i8);
        let parsed = Error::parse_error_msg(&msg);
        assert_eq!(parsed, "test error");
    }

    #[test]
    #[cfg(feature = "mock")]
    fn test_from_ffi() {
        let msg = b"test\0".map(|b| b as i8);
        let err = Error::from_ffi(trtx_sys::TRTX_ERROR_INVALID_ARGUMENT as i32, &msg);
        match err {
            Error::InvalidArgument(s) => assert_eq!(s, "test"),
            _ => panic!("Wrong error type"),
        }
    }
}
