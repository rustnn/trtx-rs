//! Safe Rust bindings to NVIDIA TensorRT-RTX
//!
//! ⚠️ **EXPERIMENTAL - NOT FOR PRODUCTION USE**
//!
//! This crate is in early experimental development. The API is unstable and will change.
//! This is NOT production-ready software. Use at your own risk.
//!
//! This crate provides safe, ergonomic Rust bindings to the TensorRT-RTX library
//! for high-performance deep learning inference on NVIDIA GPUs.
//!
//! # Overview
//!
//! TensorRT-RTX enables efficient inference by:
//! - Optimizing neural network graphs
//! - Fusing layers and operations
//! - Selecting optimal kernels for your hardware
//! - Supporting dynamic shapes and batching
//!
//! # Workflow
//!
//! Using TensorRT-RTX typically follows two phases:
//!
//! ## Build Phase (Ahead-of-Time)
//!
//! 1. Create a [`Logger`] to capture TensorRT messages
//! 2. Create a [`Builder`] to construct an optimized engine
//! 3. Define your network using [`NetworkDefinition`]
//! 4. Configure optimization with [`BuilderConfig`]
//! 5. Build and serialize the engine to disk
//!
//! ## Inference Phase (Runtime)
//!
//! 1. Create a [`Runtime`] with a logger
//! 2. Deserialize the engine using [`Runtime::deserialize_cuda_engine`]
//! 3. Create an [`ExecutionContext`] from the engine
//! 4. Bind input/output tensors
//! 5. Execute inference with [`ExecutionContext::enqueue_v3`]
//!
//! # Example
//!
//! ```rust,no_run
//! use trtx::{Logger, Builder, Runtime};
//! use trtx::builder::{BuilderConfig, MemoryPoolType, network_flags};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create logger
//! let logger = Logger::stderr()?;
//!
//! // Build phase
//! let builder = Builder::new(&logger)?;
//! let mut network = builder.create_network(network_flags::EXPLICIT_BATCH)?;
//! let mut config = builder.create_config()?;
//!
//! // Configure memory
//! config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 30)?;
//!
//! // Build and serialize
//! let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
//! std::fs::write("model.engine", &engine_data)?;
//!
//! // Inference phase
//! let runtime = Runtime::new(&logger)?;
//! let engine = runtime.deserialize_cuda_engine(&engine_data)?;
//! let context = engine.create_execution_context()?;
//!
//! // List I/O tensors
//! let num_tensors = engine.get_nb_io_tensors()?;
//! for i in 0..num_tensors {
//!     let name = engine.get_tensor_name(i)?;
//!     println!("Tensor {}: {}", i, name);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Safety
//!
//! This crate provides safe abstractions over the underlying C++ API. However,
//! some operations (like setting tensor addresses and enqueueing inference)
//! require careful management of CUDA memory and are marked as `unsafe`.
//!
//! # Prerequisites
//!
//! - NVIDIA TensorRT-RTX library installed
//! - CUDA Runtime
//! - Compatible NVIDIA GPU
//!
//! Set the `TENSORRT_RTX_DIR` environment variable to the installation path
//! if TensorRT-RTX is not in a standard location.

// Allow unnecessary casts - they're needed for real mode (u32) but not mock mode (i32)
#![cfg_attr(feature = "mock", allow(clippy::unnecessary_cast))]

pub mod autocxx_helpers;
pub mod builder;
pub mod cuda;
pub mod enum_helpers;
pub mod error;
pub mod executor;
pub mod logger;
pub mod network;
pub mod onnx_parser;
pub mod runtime;

// Re-export commonly used types
pub use builder::{Builder, BuilderConfig};
pub use cuda::{synchronize, DeviceBuffer};
pub use enum_helpers::{
    activation_type_name, datatype_name, elementwise_op_name, pooling_type_name, reduce_op_name,
    unary_op_name,
};
pub use error::{Error, Result};
pub use executor::{run_onnx_with_tensorrt, run_onnx_zeroed, TensorInput, TensorOutput};
pub use logger::{LogHandler, Logger, Severity, StderrLogger};
pub use network::{NetworkDefinition, Tensor};
pub use onnx_parser::OnnxParser;
pub use runtime::{CudaEngine, ExecutionContext, Runtime};

// Re-export TensorRT operation enums
pub use trtx_sys::nvinfer1::{
    ActivationType, DataType, ElementWiseOperation, MatrixOperation, PoolingType, ReduceOperation,
    UnaryOperation,
};
