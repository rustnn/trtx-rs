//! Error types for TensorRT-RTX operations (Rust-only; no single TensorRT C++ counterpart).

use std::ffi::NulError;

use thiserror::Error;
use trtx_sys::LayerType;

/// Result type for TensorRT-RTX operations
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Eq, PartialEq)]
pub enum PropertySetAttempt {
    SerializationFlag,
    OptimizationProfileSetDimensions,
    OptimizationProfileSetExtraMemoryTarget,
    OptimizationProfileSetShapeValues,
    BuilderConfigTacticSources,
    BuilderConfigTilingOptimizationLevel,
    BuilderConfigL2LimitForTiling,
    BuilderConfigNbComputeCapabilities,
    BuilderConfigComputeCapability,
    ExecutionContextTensorDebugState,
    DequantizeLayerBlockShape,
    QuantizeLayerBlockShape,
    AttentionLayerInput,
    AttentionLayerNumRanks,
    AttentionLayerName,
    AttentionLayerQuantizeToType,
    AttentionLayerQuantizeScale,
    AttentionLayerMetadata,
    AttentionLayerMask,
    AttentionLayerCausal,
    AttentionLayerNormalizationOp,
    AttentionLayerDecomposable,
}

// can be replaced once https://github.com/rust-lang/rust/issues/142748 becomes stable
pub(crate) trait OkOrFailedSettingProperty: Into<bool> {
    fn ok_or_err(self, e: PropertySetAttempt) -> Result<()> {
        if self.into() {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(e))
        }
    }
}

impl OkOrFailedSettingProperty for bool {}

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

    #[error("Failed to create layer: {0:?}")]
    LayerCreationFailed(LayerType),

    #[error("Failed to get generic layer from network")]
    GetLayerFailed,

    #[error("Failed to get a tensor from the network")]
    GetTensorFailed,

    #[error("Failed to create BuilderConfig")]
    BuilderConfigCreationFailed,

    #[error("Failed to set property: {0:?}")]
    FailedToSetProperty(PropertySetAttempt),
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Error::LockPoisining
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
}
