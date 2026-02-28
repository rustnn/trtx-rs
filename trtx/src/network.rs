//! Network definition for building TensorRT engines
//!
//! Types and implementations. Real/mock impls live in real/ and mock/ folders.

#[cfg(not(feature = "mock"))]
use std::pin::Pin;
#[cfg(not(feature = "mock"))]
use std::sync::Mutex;

#[cfg(not(feature = "mock"))]
use trtx_sys::nvinfer1::{
    self, IActivationLayer, ICastLayer, IConcatenationLayer, IConstantLayer, IConvolutionLayer,
    ICumulativeLayer, IDeconvolutionLayer, IDequantizeLayer, IElementWiseLayer, IGatherLayer,
    IIdentityLayer, IMatrixMultiplyLayer, IPaddingLayer, IPoolingLayer, IQuantizeLayer,
    IReduceLayer, IResizeLayer, IScaleLayer, IScatterLayer, ISelectLayer, IShuffleLayer,
    ISliceLayer, ISoftMaxLayer, ITopKLayer, IUnaryLayer,
};

use crate::error::Result;
pub use crate::real::network::NetworkDefinition;

/// Kernel and optional bias weights for convolution and deconvolution layers.
#[derive(Clone)]
pub struct ConvWeights<'a> {
    pub kernel_weights: &'a [u8],
    pub kernel_dtype: crate::DataType,
    pub bias_weights: Option<&'a [u8]>,
    pub bias_dtype: Option<crate::DataType>,
}

/// Tensor handle (opaque pointer)
#[cfg(not(feature = "mock"))]
pub struct Tensor<'network> {
    pub(crate) inner: Mutex<Pin<&'network mut nvinfer1::ITensor>>,
}
#[cfg(not(feature = "mock"))]
impl Tensor<'_> {}

#[cfg(feature = "mock")]
pub struct Tensor {}

/// Base trait for all layer types
pub trait Layer {
    /// Get the output tensor at the specified index
    #[cfg(not(feature = "mock"))]
    fn get_output(&self, index: i32) -> Result<Tensor<'_>>;

    #[cfg(feature = "mock")]
    fn get_output(&self, index: i32) -> Result<Tensor>;

    fn set_layer_name(&mut self, name: &str) -> Result<()>;
}

/// Macro to define layer struct
macro_rules! define_network_layer {
    ($name:ident, $iface:ident) => {
        #[cfg(not(feature = "mock"))]
        pub struct $name<'network> {
            pub(crate) inner: Mutex<Pin<&'network mut $iface>>,
        }
        #[cfg(not(feature = "mock"))]
        impl Drop for $name<'_> {
            fn drop(&mut self) {
                unsafe { std::ptr::drop_in_place(self.inner.get_mut().unwrap()) }
            }
        }

        #[cfg(feature = "mock")]
        pub struct $name {}

        #[cfg(not(feature = "mock"))]
        impl<'network> $name<'network> {
            pub(crate) fn from_ptr(ptr: &'network mut $iface) -> Self {
                Self {
                    inner: unsafe { Mutex::new(Pin::new_unchecked(ptr)) },
                }
            }
        }
        #[cfg(feature = "mock")]
        impl $name {
            pub(crate) fn from_ptr(_ptr: *mut std::ffi::c_void) -> Self {
                Self {}
            }
        }
    };
}

define_network_layer!(ShuffleLayer, IShuffleLayer);
define_network_layer!(ActivationLayer, IActivationLayer);
define_network_layer!(ElementWiseLayer, IElementWiseLayer);
define_network_layer!(ResizeLayer, IResizeLayer);
define_network_layer!(TopKLayer, ITopKLayer);
define_network_layer!(GatherLayer, IGatherLayer);
define_network_layer!(ScatterLayer, IScatterLayer);
define_network_layer!(SelectLayer, ISelectLayer);
define_network_layer!(MatrixMultiplyLayer, IMatrixMultiplyLayer);
define_network_layer!(SoftMaxLayer, ISoftMaxLayer);
define_network_layer!(ReduceLayer, IReduceLayer);
define_network_layer!(CumulativeLayer, ICumulativeLayer);
define_network_layer!(PoolingLayer, IPoolingLayer);
define_network_layer!(ConvolutionLayer, IConvolutionLayer);
define_network_layer!(DeconvolutionLayer, IDeconvolutionLayer);
define_network_layer!(QuantizeLayer, IQuantizeLayer);
define_network_layer!(DequantizeLayer, IDequantizeLayer);
define_network_layer!(ConstantLayer, IConstantLayer);
define_network_layer!(ConcatenationLayer, IConcatenationLayer);
define_network_layer!(ScaleLayer, IScaleLayer);
define_network_layer!(SliceLayer, ISliceLayer);
define_network_layer!(UnaryLayer, IUnaryLayer);
define_network_layer!(IdentityLayer, IIdentityLayer);
define_network_layer!(PaddingLayer, IPaddingLayer);
define_network_layer!(CastLayer, ICastLayer);

// Those are not actual ILayer in TRT
#[cfg(not(feature = "mock"))]
pub struct Loop<'network> {
    pub(crate) _inner: Mutex<Pin<&'network mut nvinfer1::ILoop>>,
}
#[cfg(not(feature = "mock"))]
pub struct IfConditional<'network> {
    pub(crate) _inner: Mutex<Pin<&'network mut nvinfer1::IIfConditional>>,
}

#[cfg(feature = "mock")]
pub struct NetworkDefinition {}
