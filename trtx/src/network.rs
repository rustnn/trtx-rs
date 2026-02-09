//! Network definition for building TensorRT engines
//!
//! Types and implementations. Real/mock impls live in real/ and mock/ folders.

use crate::error::Result;

/// Tensor handle (opaque pointer)
pub struct Tensor {
    pub(crate) inner: *mut std::ffi::c_void,
}

/// Base trait for all layer types
pub trait Layer {
    /// Get the output tensor at the specified index
    fn get_output(&self, index: i32) -> Result<Tensor>;

    /// Get the raw layer pointer (for internal use)
    fn as_ptr(&self) -> *mut std::ffi::c_void;
}

/// Macro to define layer struct
macro_rules! define_network_layer {
    ($name:ident) => {
        pub struct $name {
            pub(crate) inner: *mut std::ffi::c_void,
        }

        impl $name {
            pub(crate) fn from_ptr(ptr: *mut std::ffi::c_void) -> Self {
                Self { inner: ptr }
            }
        }
    };
}

define_network_layer!(ShuffleLayer);
define_network_layer!(ActivationLayer);
define_network_layer!(ElementWiseLayer);
define_network_layer!(ResizeLayer);
define_network_layer!(TopKLayer);
define_network_layer!(GatherLayer);
define_network_layer!(ScatterLayer);
define_network_layer!(SelectLayer);
define_network_layer!(MatrixMultiplyLayer);
define_network_layer!(SoftMaxLayer);
define_network_layer!(ReduceLayer);
define_network_layer!(CumulativeLayer);
define_network_layer!(PoolingLayer);
define_network_layer!(ConvolutionLayer);
define_network_layer!(DeconvolutionLayer);
define_network_layer!(QuantizeLayer);
define_network_layer!(DequantizeLayer);
define_network_layer!(ConstantLayer);
define_network_layer!(ConcatenationLayer);
define_network_layer!(ScaleLayer);
define_network_layer!(SliceLayer);
define_network_layer!(UnaryLayer);
define_network_layer!(IdentityLayer);
define_network_layer!(PaddingLayer);
define_network_layer!(CastLayer);

/// Network definition for building TensorRT engines
pub struct NetworkDefinition {
    pub(crate) inner: *mut std::ffi::c_void,
}

// Load implementations from real or mock
#[cfg(not(feature = "mock"))]
use crate::real::network;
#[cfg(feature = "mock")]
use crate::mock::network;
