//! Raw FFI bindings to NVIDIA TensorRT-RTX using autocxx
//!
//! ⚠️ **EXPERIMENTAL - NOT FOR PRODUCTION USE**
//!
//! This crate is in early experimental development. The API is unstable and will change.
//! This is NOT production-ready software. Use at your own risk.
//!
//! This crate provides low-level, unsafe bindings to the TensorRT-RTX C++ library.
//! For safe, ergonomic Rust API, use the `trtx` crate instead.
//!
//! # Architecture
//!
//! This crate uses a hybrid approach:
//! - **autocxx** for direct C++ bindings to TensorRT classes
//! - **Minimal C wrapper** for Logger callbacks (virtual methods)
//!
//! # Safety
//!
//! All functions in this crate are `unsafe` as they directly call into C++ code
//! and perform no safety checks. Callers must ensure:
//!
//! - Pointers are valid and properly aligned
//! - Lifetimes are managed correctly
//! - Thread safety requirements are met
//! - CUDA context is properly initialized

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

#[allow(warnings)]
mod enums {
    include!(concat!(env!("OUT_DIR"), "/enums.rs"));
}

macro_rules! better_enum {
    ($to:ident) => {
        pub use crate::enums::$to;
        impl Into<crate::nvinfer1::$to> for $to {
            fn into(self) -> crate::nvinfer1::$to {
                unsafe { transmute(self) }
            }
        }
        impl From<crate::nvinfer1::$to> for $to {
            fn from(value: crate::nvinfer1::$to) -> Self {
                unsafe { transmute(value) }
            }
        }
    };
}

use std::mem::transmute;
use std::pin::Pin;
better_enum!(LayerType);
better_enum!(ActivationType);
better_enum!(DataType);
better_enum!(ProfilingVerbosity);
better_enum!(MemoryPoolType);
better_enum!(DeviceType);
better_enum!(EngineCapability);
better_enum!(BuilderFlag);
better_enum!(PreviewFeature);
better_enum!(HardwareCompatibilityLevel);
better_enum!(RuntimePlatform);
better_enum!(TilingOptimizationLevel);
better_enum!(ComputeCapability);
better_enum!(CumulativeOperation);
better_enum!(ElementWiseOperation);
better_enum!(GatherMode);
better_enum!(InterpolationMode);
better_enum!(MatrixOperation);
better_enum!(PoolingType);
better_enum!(ReduceOperation);
better_enum!(ResizeCoordinateTransformation);
better_enum!(ResizeSelector);
better_enum!(ResizeRoundMode);
better_enum!(ScaleMode);
better_enum!(ScatterMode);
better_enum!(UnaryOperation);
better_enum!(TopKOperation);
better_enum!(LayerInformationFormat);
better_enum!(TensorLocation);
better_enum!(SerializationFlag);
better_enum!(OptProfileSelector);
better_enum!(AttentionNormalizationOp);
better_enum!(SeekPosition);
better_enum!(WeightsRole);
pub use enums::ErrorCode;

use autocxx::prelude::*;

include_cpp! {
    #include "NvInfer.h"
    #include "NvInferRuntime.h"
    #include "NvOnnxParser.h"

    safety!(unsafe_ffi)

    // Core TensorRT types
    generate!("nvinfer1::IBuilder")
    generate!("nvinfer1::IBuilderConfig")
    generate!("nvinfer1::INetworkDefinition")
    generate!("nvinfer1::ITensor")
    generate!("nvinfer1::ILayer")
    generate!("nvinfer1::IVersionedInterface")
    generate!("nvinfer1::IProgressMonitor")
    generate!("nvinfer1::IStreamWriter")
    generate!("nvinfer1::IStreamReaderV2")
    generate!("nvinfer1::IErrorRecorder")
    generate!("nvinfer1::IProfiler")
    generate!("nvinfer1::IGpuAllocator")
    generate!("nvinfer1::IDebugListener")
    generate!("nvinfer1::ISerializationConfig")
    generate!("nvinfer1::IOptimizationProfile")
    generate!("nvinfer1::IRefitter")

    // Derived layer types - for inheritance support
    generate!("nvinfer1::IActivationLayer")
    generate!("nvinfer1::IConvolutionLayer")
    generate!("nvinfer1::IPoolingLayer")
    generate!("nvinfer1::IElementWiseLayer")
    generate!("nvinfer1::IShuffleLayer")
    generate!("nvinfer1::IConcatenationLayer")
    generate!("nvinfer1::IMatrixMultiplyLayer")
    generate!("nvinfer1::IConstantLayer")
    generate!("nvinfer1::ISoftMaxLayer")
    generate!("nvinfer1::IScaleLayer")
    generate!("nvinfer1::IReduceLayer")
    generate!("nvinfer1::ISliceLayer")
    generate!("nvinfer1::IResizeLayer")
    generate!("nvinfer1::ITopKLayer")
    generate!("nvinfer1::IGatherLayer")
    generate!("nvinfer1::IScatterLayer")
    generate!("nvinfer1::ISelectLayer")
    generate!("nvinfer1::IUnaryLayer")
    generate!("nvinfer1::IIdentityLayer")
    generate!("nvinfer1::IPaddingLayer")
    generate!("nvinfer1::ICastLayer")
    generate!("nvinfer1::IDeconvolutionLayer")
    generate!("nvinfer1::IQuantizeLayer")
    generate!("nvinfer1::IDequantizeLayer")
    generate!("nvinfer1::IAssertionLayer")
    generate!("nvinfer1::ICumulativeLayer")
    generate!("nvinfer1::ILoop")
    generate!("nvinfer1::IIfConditional")
    generate!("nvinfer1::INormalizationLayer")
    generate!("nvinfer1::ISqueezeLayer")
    generate!("nvinfer1::IUnsqueezeLayer")
    generate!("nvinfer1::ILRNLayer")
    generate!("nvinfer1::IShapeLayer")
    generate!("nvinfer1::IParametricReLULayer")
    generate!("nvinfer1::IFillLayer")
    generate!("nvinfer1::IEinsumLayer")
    generate!("nvinfer1::IOneHotLayer")
    generate!("nvinfer1::INonZeroLayer")
    generate!("nvinfer1::IGridSampleLayer")
    generate!("nvinfer1::INMSLayer")
    generate!("nvinfer1::IReverseSequenceLayer")
    generate!("nvinfer1::IDynamicQuantizeLayer")
    generate!("nvinfer1::IRotaryEmbeddingLayer")
    generate!("nvinfer1::IKVCacheUpdateLayer")
    generate!("nvinfer1::IRaggedSoftMaxLayer")
    generate!("nvinfer1::ILoopBoundaryLayer")
    generate!("nvinfer1::IRecurrenceLayer")
    generate!("nvinfer1::ILoopOutputLayer")
    generate!("nvinfer1::ITripLimitLayer")
    generate!("nvinfer1::IIteratorLayer")
    generate!("nvinfer1::IConditionLayer")
    generate!("nvinfer1::IIfConditionalOutputLayer")
    generate!("nvinfer1::IIfConditionalInputLayer")
    generate!("nvinfer1::IAttentionBoundaryLayer")
    generate!("nvinfer1::IAttentionInputLayer")
    generate!("nvinfer1::IAttentionOutputLayer")
    generate!("nvinfer1::IAttention")
    // NOTE: IRNNv2Layer is deprecated (TRT_DEPRECATED) and autocxx cannot generate bindings for it
    // RNN operations (lstm, lstmCell, gru, gruCell) remain deferred until we can work around this
    // generate!("nvinfer1::IRNNv2Layer")

    generate!("nvinfer1::IRuntime")
    generate!("nvinfer1::ICudaEngine")
    generate!("nvinfer1::IExecutionContext")
    generate!("nvinfer1::IEngineInspector")
    generate!("nvinfer1::IHostMemory")
    generate!("nvinfer1::LayerInformationFormat")

    // Try generating Dims64 directly (base class, not the typedef alias)
    generate_pod!("nvinfer1::Dims64")

    generate_pod!("nvinfer1::DataType")
    generate_pod!("nvinfer1::TensorIOMode")
    generate_pod!("nvinfer1::MemoryPoolType")
    generate_pod!("nvinfer1::NetworkDefinitionCreationFlag")
    generate_pod!("nvinfer1::ActivationType")
    generate_pod!("nvinfer1::PoolingType")
    generate_pod!("nvinfer1::ElementWiseOperation")
    generate_pod!("nvinfer1::MatrixOperation")
    generate_pod!("nvinfer1::UnaryOperation")
    generate_pod!("nvinfer1::ReduceOperation")
    generate_pod!("nvinfer1::CumulativeOperation")
    generate_pod!("nvinfer1::GatherMode")
    generate_pod!("nvinfer1::ScatterMode")
    generate_pod!("nvinfer1::InterpolationMode")
    generate_pod!("nvinfer1::ResizeCoordinateTransformation")
    generate_pod!("nvinfer1::ResizeSelector")
    generate_pod!("nvinfer1::ResizeRoundMode")
    generate_pod!("nvinfer1::ProfilingVerbosity")
    generate_pod!("nvinfer1::EngineCapability")
    generate_pod!("nvinfer1::BuilderFlag")
    generate_pod!("nvinfer1::BuilderFlags")
    generate_pod!("nvinfer1::DeviceType")
    generate_pod!("nvinfer1::TacticSource")
    generate_pod!("nvinfer1::TacticSources")
    generate_pod!("nvinfer1::PreviewFeature")
    generate_pod!("nvinfer1::HardwareCompatibilityLevel")
    generate_pod!("nvinfer1::RuntimePlatform")
    generate_pod!("nvinfer1::TilingOptimizationLevel")
    generate_pod!("nvinfer1::ComputeCapability")
    generate_pod!("nvinfer1::APILanguage")
    // NOTE: RNN enums commented out because IRNNv2Layer (deprecated) cannot be generated
    // generate!("nvinfer1::RNNOperation")
    // generate!("nvinfer1::RNNDirection")
    // generate!("nvinfer1::RNNInputMode")
    // generate!("nvinfer1::RNNGateType")
    generate_pod!("nvinfer1::Weights")
    generate_pod!("nvinfer1::Permutation")
    generate_pod!("nvinfer1::TripLimit")
    generate_pod!("nvinfer1::LoopOutput")
    generate_pod!("nvinfer1::AttentionNormalizationOp")
    generate_pod!("nvinfer1::WeightsRole")

    generate!("nvinfer1::ErrorCode")
    generate!("nvinfer1::LayerType")
    generate!("nvinfer1::SerializationFlags")
    generate!("nvinfer1::SerializationFlag")
    generate!("nvinfer1::OptProfileSelector")

    // NOTE: createInferBuilder/Runtime moved to logger_bridge.cpp (autocxx struggles with these)

    // ONNX Parser
    generate!("nvonnxparser::IParser")
    // NOTE: createParser also moved to logger_bridge.cpp

}

pub unsafe trait TrtLayer {
    const TYPE: LayerType;
    fn as_layer(&self) -> &nvinfer1::ILayer {
        // can't use safe `as_ref() -> &nvinfer1::ILayer` because only implemented for direct
        // subclasses of ILayer
        unsafe {
            (self as *const Self as *const nvinfer1::ILayer)
                .as_ref()
                .unwrap()
        }
    }
    fn as_layer_pin_mut(&mut self) -> Pin<&mut nvinfer1::ILayer> {
        unsafe {
            Pin::new_unchecked(
                (self as *mut Self as *mut nvinfer1::ILayer)
                    .as_mut()
                    .unwrap(),
            )
        }
    }
}

unsafe impl TrtLayer for nvinfer1::IActivationLayer {
    const TYPE: LayerType = LayerType::kACTIVATION;
}
unsafe impl TrtLayer for nvinfer1::IConvolutionLayer {
    const TYPE: LayerType = LayerType::kCONVOLUTION;
}
unsafe impl TrtLayer for nvinfer1::ICastLayer {
    const TYPE: LayerType = LayerType::kCAST;
}
unsafe impl TrtLayer for nvinfer1::IPoolingLayer {
    const TYPE: LayerType = LayerType::kPOOLING;
}
unsafe impl TrtLayer for nvinfer1::ILRNLayer {
    const TYPE: LayerType = LayerType::kLRN;
}
unsafe impl TrtLayer for nvinfer1::IScaleLayer {
    const TYPE: LayerType = LayerType::kSCALE;
}
unsafe impl TrtLayer for nvinfer1::ISoftMaxLayer {
    const TYPE: LayerType = LayerType::kSOFTMAX;
}
unsafe impl TrtLayer for nvinfer1::IDeconvolutionLayer {
    const TYPE: LayerType = LayerType::kDECONVOLUTION;
}
unsafe impl TrtLayer for nvinfer1::IConcatenationLayer {
    const TYPE: LayerType = LayerType::kCONCATENATION;
}
unsafe impl TrtLayer for nvinfer1::IElementWiseLayer {
    const TYPE: LayerType = LayerType::kELEMENTWISE;
}
unsafe impl TrtLayer for nvinfer1::IUnaryLayer {
    const TYPE: LayerType = LayerType::kUNARY;
}
unsafe impl TrtLayer for nvinfer1::IPaddingLayer {
    const TYPE: LayerType = LayerType::kPADDING;
}
unsafe impl TrtLayer for nvinfer1::IShuffleLayer {
    const TYPE: LayerType = LayerType::kSHUFFLE;
}
unsafe impl TrtLayer for nvinfer1::IReduceLayer {
    const TYPE: LayerType = LayerType::kREDUCE;
}
unsafe impl TrtLayer for nvinfer1::ITopKLayer {
    const TYPE: LayerType = LayerType::kTOPK;
}
unsafe impl TrtLayer for nvinfer1::IGatherLayer {
    const TYPE: LayerType = LayerType::kGATHER;
}
unsafe impl TrtLayer for nvinfer1::IMatrixMultiplyLayer {
    const TYPE: LayerType = LayerType::kMATRIX_MULTIPLY;
}
unsafe impl TrtLayer for nvinfer1::IRaggedSoftMaxLayer {
    const TYPE: LayerType = LayerType::kRAGGED_SOFTMAX;
}
unsafe impl TrtLayer for nvinfer1::IConstantLayer {
    const TYPE: LayerType = LayerType::kCONSTANT;
}
unsafe impl TrtLayer for nvinfer1::IIdentityLayer {
    const TYPE: LayerType = LayerType::kIDENTITY;
}
unsafe impl TrtLayer for nvinfer1::ISliceLayer {
    const TYPE: LayerType = LayerType::kSLICE;
}
unsafe impl TrtLayer for nvinfer1::IShapeLayer {
    const TYPE: LayerType = LayerType::kSHAPE;
}
unsafe impl TrtLayer for nvinfer1::IParametricReLULayer {
    const TYPE: LayerType = LayerType::kPARAMETRIC_RELU;
}
unsafe impl TrtLayer for nvinfer1::IResizeLayer {
    const TYPE: LayerType = LayerType::kRESIZE;
}
unsafe impl TrtLayer for nvinfer1::ISelectLayer {
    const TYPE: LayerType = LayerType::kSELECT;
}
unsafe impl TrtLayer for nvinfer1::IFillLayer {
    const TYPE: LayerType = LayerType::kFILL;
}
unsafe impl TrtLayer for nvinfer1::IQuantizeLayer {
    const TYPE: LayerType = LayerType::kQUANTIZE;
}
unsafe impl TrtLayer for nvinfer1::IDequantizeLayer {
    const TYPE: LayerType = LayerType::kDEQUANTIZE;
}
unsafe impl TrtLayer for nvinfer1::IScatterLayer {
    const TYPE: LayerType = LayerType::kSCATTER;
}
unsafe impl TrtLayer for nvinfer1::IEinsumLayer {
    const TYPE: LayerType = LayerType::kEINSUM;
}
unsafe impl TrtLayer for nvinfer1::IAssertionLayer {
    const TYPE: LayerType = LayerType::kASSERTION;
}
unsafe impl TrtLayer for nvinfer1::IOneHotLayer {
    const TYPE: LayerType = LayerType::kONE_HOT;
}
unsafe impl TrtLayer for nvinfer1::INonZeroLayer {
    const TYPE: LayerType = LayerType::kNON_ZERO;
}
unsafe impl TrtLayer for nvinfer1::IGridSampleLayer {
    const TYPE: LayerType = LayerType::kGRID_SAMPLE;
}
unsafe impl TrtLayer for nvinfer1::INMSLayer {
    const TYPE: LayerType = LayerType::kNMS;
}
unsafe impl TrtLayer for nvinfer1::IReverseSequenceLayer {
    const TYPE: LayerType = LayerType::kREVERSE_SEQUENCE;
}
unsafe impl TrtLayer for nvinfer1::INormalizationLayer {
    const TYPE: LayerType = LayerType::kNORMALIZATION;
}
unsafe impl TrtLayer for nvinfer1::ISqueezeLayer {
    const TYPE: LayerType = LayerType::kSQUEEZE;
}
unsafe impl TrtLayer for nvinfer1::IUnsqueezeLayer {
    const TYPE: LayerType = LayerType::kUNSQUEEZE;
}
unsafe impl TrtLayer for nvinfer1::ICumulativeLayer {
    const TYPE: LayerType = LayerType::kCUMULATIVE;
}
unsafe impl TrtLayer for nvinfer1::IDynamicQuantizeLayer {
    const TYPE: LayerType = LayerType::kDYNAMIC_QUANTIZE;
}
unsafe impl TrtLayer for nvinfer1::IRotaryEmbeddingLayer {
    const TYPE: LayerType = LayerType::kROTARY_EMBEDDING;
}
unsafe impl TrtLayer for nvinfer1::IKVCacheUpdateLayer {
    const TYPE: LayerType = LayerType::kKVCACHE_UPDATE;
}

// indirect subclasses of ILayer e.g. via ILoopBoundaryLayer, IAttentionBoundaryLayer, IIfConditionalBoundaryLayer

unsafe impl TrtLayer for nvinfer1::IAttentionInputLayer {
    const TYPE: LayerType = LayerType::kATTENTION_INPUT;
}
unsafe impl TrtLayer for nvinfer1::IAttentionOutputLayer {
    const TYPE: LayerType = LayerType::kATTENTION_OUTPUT;
}
unsafe impl TrtLayer for nvinfer1::ILoopBoundaryLayer {
    const TYPE: LayerType = LayerType::kTRIP_LIMIT;
}
unsafe impl TrtLayer for nvinfer1::ILoopOutputLayer {
    const TYPE: LayerType = LayerType::kLOOP_OUTPUT;
}
unsafe impl TrtLayer for nvinfer1::IRecurrenceLayer {
    const TYPE: LayerType = LayerType::kRECURRENCE;
}
unsafe impl TrtLayer for nvinfer1::ITripLimitLayer {
    const TYPE: LayerType = LayerType::kTRIP_LIMIT;
}
unsafe impl TrtLayer for nvinfer1::IIteratorLayer {
    const TYPE: LayerType = LayerType::kITERATOR;
}
unsafe impl TrtLayer for nvinfer1::IConditionLayer {
    const TYPE: LayerType = LayerType::kCONDITION;
}
unsafe impl TrtLayer for nvinfer1::IIfConditionalOutputLayer {
    const TYPE: LayerType = LayerType::kCONDITIONAL_OUTPUT;
}
unsafe impl TrtLayer for nvinfer1::IIfConditionalInputLayer {
    const TYPE: LayerType = LayerType::kCONDITIONAL_INPUT;
}
unsafe impl TrtLayer for nvinfer1::IAttentionBoundaryLayer {
    const TYPE: LayerType = LayerType::kATTENTION_INPUT;
}

// Logger bridge C functions
unsafe extern "C" {
    pub unsafe fn get_tensorrt_version() -> u32;
    pub unsafe fn create_rust_logger_bridge(
        callback: RustLogCallback,
        user_data: *mut std::ffi::c_void,
    ) -> *mut RustLoggerBridge;

    pub unsafe fn destroy_rust_logger_bridge(logger: *mut RustLoggerBridge);

    pub unsafe fn get_logger_interface(logger: *mut RustLoggerBridge) -> *mut std::ffi::c_void; // Returns ILogger*
                                                                                                //
    pub unsafe fn trtx_create_progress_monitor_subclass(
        user_data: *mut std::ffi::c_void,
        phaseStart: unsafe extern "system" fn(
            user_data: *mut std::ffi::c_void,
            phaseName: *const ::std::os::raw::c_char,
            parentPhase: *const ::std::os::raw::c_char,
            nbSteps: i32,
        ),
        stepComplete: unsafe extern "system" fn(
            user_data: *mut std::ffi::c_void,
            phaseName: *const ::std::os::raw::c_char,
            step: i32,
        ) -> bool,
        phaseFinish: unsafe extern "system" fn(
            user_data: *mut std::ffi::c_void,
            phaseName: *const ::std::os::raw::c_char,
        ),
    ) -> *mut std::ffi::c_void;
    pub unsafe fn trtx_destroy_progress_monitor_subclass(cpp_obj: *mut std::ffi::c_void);
    pub unsafe fn trtx_create_gpu_allocator_subclass(
        rust_impl: *mut std::ffi::c_void,
        allocateAsync: *mut std::ffi::c_void,
        reallocate: *mut std::ffi::c_void,
        deallocateAsync: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;
    pub unsafe fn trtx_destroy_gpu_allocator_subclass(cpp_obj: *mut std::ffi::c_void);
    pub unsafe fn trtx_create_error_recorder_subclass(
        rust_impl: *mut std::ffi::c_void,
        getNbErrors: *mut std::ffi::c_void,
        getErrorCode: *mut std::ffi::c_void,
        getErrorDesc: *mut std::ffi::c_void,
        hasOverflowed: *mut std::ffi::c_void,
        clear: *mut std::ffi::c_void,
        reportError: *mut std::ffi::c_void,
        incRefCount: *mut std::ffi::c_void,
        decRefCount: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;
    pub unsafe fn trtx_destroy_error_recorder_subclass(cpp_obj: *mut std::ffi::c_void);

    // TensorRT factory functions (wrapped as simple C functions)
    #[cfg(feature = "link_tensorrt_rtx")]
    pub unsafe fn create_infer_builder(logger: *mut std::ffi::c_void) -> *mut std::ffi::c_void; // Returns IBuilder*

    #[cfg(feature = "link_tensorrt_rtx")]
    pub unsafe fn create_infer_runtime(logger: *mut std::ffi::c_void) -> *mut std::ffi::c_void; // Returns IRuntime*

    #[cfg(feature = "link_tensorrt_rtx")]
    pub fn create_infer_refitter(
        cuda_engine: *mut std::ffi::c_void,
        logger: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void; // Returns IRefitter*

    pub unsafe fn trtx_refitter_get_missing(
        refitter: *mut std::ffi::c_void,
        size: i32,
        layer_names: *mut *const std::os::raw::c_char,
        roles: *mut i32,
    ) -> i32;

    pub unsafe fn trtx_refitter_get_all(
        refitter: *mut std::ffi::c_void,
        size: i32,
        layer_names: *mut *const std::os::raw::c_char,
        roles: *mut i32,
    ) -> i32;

    pub unsafe fn trtx_refitter_get_missing_weights(
        refitter: *mut std::ffi::c_void,
        size: i32,
        weights_names: *mut *const std::os::raw::c_char,
    ) -> i32;

    pub unsafe fn trtx_refitter_get_all_weights(
        refitter: *mut std::ffi::c_void,
        size: i32,
        weights_names: *mut *const std::os::raw::c_char,
    ) -> i32;

    // ONNX Parser factory function
    #[cfg(feature = "link_tensorrt_onnxparser")]
    pub unsafe fn create_onnx_parser(
        network: *mut std::ffi::c_void,
        logger: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void; // Returns IParser*
                                //
    pub unsafe fn network_add_concatenation(
        network: *mut std::ffi::c_void,
        inputs: *mut *mut std::ffi::c_void,
        nb_inputs: i32,
    ) -> *mut std::ffi::c_void;

    // Parser methods
    pub unsafe fn parser_parse(
        parser: *mut std::ffi::c_void,
        data: *const std::ffi::c_void,
        size: usize,
    ) -> bool;
    pub unsafe fn parser_get_nb_errors(parser: *mut std::ffi::c_void) -> i32;
    pub unsafe fn parser_get_error(
        parser: *mut std::ffi::c_void,
        index: i32,
    ) -> *mut std::ffi::c_void;
    pub unsafe fn parser_error_desc(error: *mut std::ffi::c_void) -> *const std::os::raw::c_char;

}

// Opaque type for logger bridge
#[repr(C)]
pub struct RustLoggerBridge {
    _unused: [u8; 0],
}

// Rust callback type for logger
pub type RustLogCallback = unsafe extern "C" fn(
    user_data: *mut std::ffi::c_void,
    severity: i32,
    msg: *const std::os::raw::c_char,
);

// Re-export TensorRT types from the private ffi module
pub mod nvinfer1 {
    pub use super::ffi::nvinfer1::*;
}

#[cfg(feature = "onnxparser")]
pub mod nvonnxparser {
    pub use super::ffi::nvonnxparser::*;
}

// Re-export Dims64 as Dims to match TensorRT's typedef
pub use nvinfer1::Dims64;
pub type Dims = Dims64;

// Re-export InterpolationMode as ResizeMode to match TensorRT's typedef
pub type ResizeMode = InterpolationMode;

/// Helper methods for Dims construction (avoiding name collision with generated constructor)
impl Dims64 {
    /// Create a Dims from a slice of dimensions
    pub fn from_slice(dims: &[i64]) -> Self {
        let mut d = [0i64; 8];
        let nb_dims = dims.len().min(8) as i32;
        d[..nb_dims as usize].copy_from_slice(&dims[..nb_dims as usize]);
        Self { nbDims: nb_dims, d }
    }

    /// Create a 2D Dims
    pub fn new_2d(d0: i64, d1: i64) -> Self {
        Self {
            nbDims: 2,
            d: [d0, d1, 0, 0, 0, 0, 0, 0],
        }
    }

    /// Create a 3D Dims
    pub fn new_3d(d0: i64, d1: i64, d2: i64) -> Self {
        Self {
            nbDims: 3,
            d: [d0, d1, d2, 0, 0, 0, 0, 0],
        }
    }

    /// Create a 4D Dims
    pub fn new_4d(d0: i64, d1: i64, d2: i64, d3: i64) -> Self {
        Self {
            nbDims: 4,
            d: [d0, d1, d2, d3, 0, 0, 0, 0],
        }
    }
}

// Re-export Weights
pub use nvinfer1::Weights;

/// Helper methods for Weights construction
impl nvinfer1::Weights {
    /// Create a Weights with FLOAT data type
    pub fn new_float(values_ptr: *const std::ffi::c_void, count_val: i64) -> Self {
        Self {
            type_: nvinfer1::DataType::kFLOAT,
            values: values_ptr,
            count: count_val,
        }
    }

    /// Create a Weights with specified data type
    pub fn new_with_type(
        data_type: nvinfer1::DataType,
        values_ptr: *const std::ffi::c_void,
        count_val: i64,
    ) -> Self {
        Self {
            type_: data_type,
            values: values_ptr,
            count: count_val,
        }
    }
}

impl DataType {
    pub const fn size_bits(self) -> usize {
        match self {
            DataType::kFLOAT => 32,
            DataType::kHALF => 16,
            DataType::kINT8 => 8,
            DataType::kINT32 => 32,
            DataType::kBOOL => 8,
            DataType::kUINT8 => 8,
            DataType::kFP8 => 8,
            DataType::kBF16 => 16,
            DataType::kINT64 => 64,
            DataType::kINT4 => 4,
            DataType::kFP4 => 4,
            DataType::kE8M0 => 8,
        }
    }
}
