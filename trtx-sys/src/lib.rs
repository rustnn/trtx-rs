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

// Mock mode uses old-style bindings
#[cfg(feature = "mock")]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Real mode uses autocxx
#[cfg(not(feature = "mock"))]
pub mod real_bindings {
    use autocxx::prelude::*;

    include_cpp! {
        #include "NvInfer.h"
        #include "NvOnnxParser.h"
        #include "cuda_runtime.h"

        safety!(unsafe_ffi)

        // Core TensorRT types
        generate!("nvinfer1::IBuilder")
        generate!("nvinfer1::IBuilderConfig")
        generate!("nvinfer1::INetworkDefinition")
        generate!("nvinfer1::ITensor")
        generate!("nvinfer1::ILayer")

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
        generate!("nvinfer1::ISelectLayer")
        generate!("nvinfer1::IUnaryLayer")
        generate!("nvinfer1::IIdentityLayer")
        generate!("nvinfer1::IPaddingLayer")
        generate!("nvinfer1::ICastLayer")
        generate!("nvinfer1::IAssertionLayer")
        generate!("nvinfer1::ILoop")
        generate!("nvinfer1::IIfConditional")

        generate!("nvinfer1::IRuntime")
        generate!("nvinfer1::ICudaEngine")
        generate!("nvinfer1::IExecutionContext")
        generate!("nvinfer1::IHostMemory")

        // Try generating Dims64 directly (base class, not the typedef alias)
        generate_pod!("nvinfer1::Dims64")

        generate!("nvinfer1::DataType")
        generate!("nvinfer1::TensorIOMode")
        generate!("nvinfer1::MemoryPoolType")
        generate!("nvinfer1::NetworkDefinitionCreationFlag")
        generate!("nvinfer1::ActivationType")
        generate!("nvinfer1::PoolingType")
        generate!("nvinfer1::ElementWiseOperation")
        generate!("nvinfer1::MatrixOperation")
        generate!("nvinfer1::UnaryOperation")
        generate!("nvinfer1::ReduceOperation")
        generate_pod!("nvinfer1::Weights")

        // NOTE: createInferBuilder/Runtime moved to logger_bridge.cpp (autocxx struggles with these)

        // ONNX Parser
        generate!("nvonnxparser::IParser")
        // NOTE: createParser also moved to logger_bridge.cpp

        // CUDA functions
        generate!("cudaMalloc")
        generate!("cudaFree")
        generate!("cudaMemcpy")
        generate!("cudaMemcpyKind")
        generate!("cudaDeviceSynchronize")
        generate!("cudaGetErrorString")
        generate!("cudaError_t")
    }

    // Logger bridge C functions
    extern "C" {
        pub fn create_rust_logger_bridge(
            callback: RustLogCallback,
            user_data: *mut std::ffi::c_void,
        ) -> *mut RustLoggerBridge;

        pub fn destroy_rust_logger_bridge(logger: *mut RustLoggerBridge);

        pub fn get_logger_interface(logger: *mut RustLoggerBridge) -> *mut std::ffi::c_void; // Returns ILogger*

        // TensorRT factory functions (wrapped as simple C functions)
        pub fn create_infer_builder(logger: *mut std::ffi::c_void) -> *mut std::ffi::c_void; // Returns IBuilder*

        pub fn create_infer_runtime(logger: *mut std::ffi::c_void) -> *mut std::ffi::c_void; // Returns IRuntime*

        // ONNX Parser factory function
        pub fn create_onnx_parser(
            network: *mut std::ffi::c_void,
            logger: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void; // Returns IParser*

        // Builder methods
        pub fn builder_create_network_v2(
            builder: *mut std::ffi::c_void,
            flags: u32,
        ) -> *mut std::ffi::c_void;

        pub fn builder_create_config(builder: *mut std::ffi::c_void) -> *mut std::ffi::c_void;

        pub fn builder_build_serialized_network(
            builder: *mut std::ffi::c_void,
            network: *mut std::ffi::c_void,
            config: *mut std::ffi::c_void,
            out_size: *mut usize,
        ) -> *mut std::ffi::c_void;

        pub fn builder_config_set_memory_pool_limit(
            config: *mut std::ffi::c_void,
            pool_type: i32,
            limit: usize,
        );

        // Network methods
        // network_add_input - REMOVED - Using direct autocxx
        // network_add_convolution - REMOVED - Using direct autocxx
        // network_add_constant - REMOVED - Using direct autocxx
        // network_add_scale - REMOVED - Using direct autocxx

        pub fn network_mark_output(
            network: *mut std::ffi::c_void,
            tensor: *mut std::ffi::c_void,
        ) -> bool;

        pub fn network_get_nb_inputs(network: *mut std::ffi::c_void) -> i32;
        pub fn network_get_nb_outputs(network: *mut std::ffi::c_void) -> i32;
        pub fn network_get_input(
            network: *mut std::ffi::c_void,
            index: i32,
        ) -> *mut std::ffi::c_void;
        pub fn network_get_output(
            network: *mut std::ffi::c_void,
            index: i32,
        ) -> *mut std::ffi::c_void;

        // network_add_activation - REMOVED - Using direct autocxx

        // network_add_pooling - REMOVED - Using direct autocxx

        // network_add_elementwise - REMOVED - Using direct autocxx

        // network_add_shuffle - REMOVED - Using direct autocxx

        pub fn network_add_concatenation(
            network: *mut std::ffi::c_void,
            inputs: *mut *mut std::ffi::c_void,
            nb_inputs: i32,
        ) -> *mut std::ffi::c_void;

        // network_add_reduce - REMOVED - Using direct autocxx

        // network_add_slice - REMOVED - Using direct autocxx

        // network_add_resize - REMOVED - Using direct autocxx

        // network_add_topk - REMOVED - Using direct autocxx

        // network_add_gather - REMOVED - Using direct autocxx

        // network_add_select - REMOVED - Using direct autocxx

        pub fn network_add_assertion(
            network: *mut std::ffi::c_void,
            condition: *mut std::ffi::c_void,
            message: *const std::os::raw::c_char,
        ) -> *mut std::ffi::c_void;

        pub fn network_add_loop(network: *mut std::ffi::c_void) -> *mut std::ffi::c_void;

        pub fn network_add_if_conditional(network: *mut std::ffi::c_void) -> *mut std::ffi::c_void;

        // Tensor methods
        pub fn tensor_get_name(tensor: *mut std::ffi::c_void) -> *const std::os::raw::c_char;
        pub fn tensor_set_name(tensor: *mut std::ffi::c_void, name: *const std::os::raw::c_char);
        pub fn tensor_get_dimensions(
            tensor: *mut std::ffi::c_void,
            dims: *mut i32,
            nb_dims: *mut i32,
        ) -> *mut std::ffi::c_void;
        pub fn tensor_get_type(tensor: *mut std::ffi::c_void) -> i32;

        // Runtime methods
        pub fn runtime_deserialize_cuda_engine(
            runtime: *mut std::ffi::c_void,
            data: *const std::ffi::c_void,
            size: usize,
        ) -> *mut std::ffi::c_void;

        // Engine methods
        pub fn engine_get_nb_io_tensors(engine: *mut std::ffi::c_void) -> i32;
        pub fn engine_get_tensor_name(
            engine: *mut std::ffi::c_void,
            index: i32,
        ) -> *const std::os::raw::c_char;
        pub fn engine_create_execution_context(
            engine: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void;

        // ExecutionContext methods
        pub fn context_set_tensor_address(
            context: *mut std::ffi::c_void,
            name: *const std::os::raw::c_char,
            data: *mut std::ffi::c_void,
        ) -> bool;
        pub fn context_enqueue_v3(
            context: *mut std::ffi::c_void,
            stream: *mut std::ffi::c_void,
        ) -> bool;

        // Parser methods
        pub fn parser_parse(
            parser: *mut std::ffi::c_void,
            data: *const std::ffi::c_void,
            size: usize,
        ) -> bool;
        pub fn parser_get_nb_errors(parser: *mut std::ffi::c_void) -> i32;
        pub fn parser_get_error(parser: *mut std::ffi::c_void, index: i32)
            -> *mut std::ffi::c_void;
        pub fn parser_error_desc(error: *mut std::ffi::c_void) -> *const std::os::raw::c_char;

        // CUDA wrappers
        pub fn cuda_malloc_wrapper(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
        pub fn cuda_free_wrapper(ptr: *mut std::ffi::c_void) -> i32;
        pub fn cuda_memcpy_wrapper(
            dst: *mut std::ffi::c_void,
            src: *const std::ffi::c_void,
            count: usize,
            kind: i32,
        ) -> i32;
        pub fn cuda_device_synchronize_wrapper() -> i32;
        pub fn cuda_get_error_string_wrapper(error: i32) -> *const std::os::raw::c_char;

        // Destruction methods
        pub fn delete_builder(builder: *mut std::ffi::c_void);
        pub fn delete_network(network: *mut std::ffi::c_void);
        pub fn delete_config(config: *mut std::ffi::c_void);
        pub fn delete_runtime(runtime: *mut std::ffi::c_void);
        pub fn delete_engine(engine: *mut std::ffi::c_void);
        pub fn delete_context(context: *mut std::ffi::c_void);
        pub fn delete_parser(parser: *mut std::ffi::c_void);
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

    pub mod nvonnxparser {
        pub use super::ffi::nvonnxparser::*;
    }

    // Re-export CUDA functions
    pub use ffi::{
        cudaDeviceSynchronize, cudaError_t, cudaFree, cudaGetErrorString, cudaMalloc, cudaMemcpy,
        cudaMemcpyKind,
    };

    // Re-export Dims64 as Dims to match TensorRT's typedef
    pub use nvinfer1::Dims64;
    pub type Dims = Dims64;

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
}

#[cfg(not(feature = "mock"))]
pub use real_bindings::*;

#[cfg(test)]
mod tests {
    #[cfg(feature = "mock")]
    use super::*;

    #[test]
    #[cfg(feature = "mock")]
    fn test_constants() {
        // Verify error codes are defined
        assert_eq!(TRTX_SUCCESS, 0);
        assert_ne!(TRTX_ERROR_INVALID_ARGUMENT, TRTX_SUCCESS);
    }
}
