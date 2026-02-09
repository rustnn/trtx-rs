//! Unified mock for TensorRT-RTX and CUDA
//!
//! This module provides a single mock implementation used when the `mock` feature
//! is enabled. Mock mode allows development without TensorRT-RTX or GPU hardware.
//! All TensorRT (builder, network, runtime, engine) and CUDA (DeviceBuffer, streams)
//! operations are stubbed here - no real GPU or TensorRT libraries are required.

pub(crate) mod builder;
pub(crate) mod cuda;
mod error;
pub(crate) mod logger;
pub(crate) mod network;
pub(crate) mod onnx_parser;
pub(crate) mod runtime;

pub(crate) use error::*;
pub(crate) use network::default_engine_tensor_shape;
