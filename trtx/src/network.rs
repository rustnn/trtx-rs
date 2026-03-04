//! Network definition for building TensorRT engines
//!
//! Types and implementations. Real/mock impls live in real/ and mock/ folders.

use std::mem::transmute;
use std::pin::Pin;
use std::sync::Mutex;

use trtx_sys::nvinfer1::{
    self, IActivationLayer, ICastLayer, IConcatenationLayer, IConstantLayer, IConvolutionLayer,
    ICumulativeLayer, IDeconvolutionLayer, IDequantizeLayer, IElementWiseLayer, IGatherLayer,
    IIdentityLayer, ILayer, IMatrixMultiplyLayer, INormalizationLayer, IPaddingLayer,
    IPoolingLayer, IQuantizeLayer, IReduceLayer, IResizeLayer, IScaleLayer, IScatterLayer,
    ISelectLayer, IShuffleLayer, ISliceLayer, ISoftMaxLayer, ISqueezeLayer, ITopKLayer,
    IUnaryLayer, IUnsqueezeLayer,
};

use crate::error::{Error, Result};
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
pub struct Tensor<'network> {
    pub(crate) inner: Mutex<Pin<&'network mut nvinfer1::ITensor>>,
}
impl Tensor<'_> {}

/// Base trait for all layer types
pub trait Layer {
    /// Get the output tensor at the specified index
    fn get_output(&self, index: i32) -> Result<Tensor<'_>>;

    fn set_layer_name(&mut self, name: &str) -> Result<()>;
}

/// Macro to define layer struct
macro_rules! define_network_layer {
    ($name:ident, $iface:ident) => {
        pub struct $name<'network> {
            pub(crate) inner: Mutex<Pin<&'network mut $iface>>,
        }
        impl Drop for $name<'_> {
            fn drop(&mut self) {
                unsafe { std::ptr::drop_in_place(self.inner.get_mut().unwrap()) }
            }
        }

        impl<'network> $name<'network> {
            #[allow(dead_code)]
            pub(crate) fn from_ptr(ptr: &'network mut $iface) -> Self {
                Self {
                    inner: unsafe { Mutex::new(Pin::new_unchecked(ptr)) },
                }
            }
        }

        impl Layer for $name<'_> {
            fn get_output(&self, index: i32) -> Result<Tensor<'_>> {
                let mut lock = self.inner.lock().unwrap();
                let ptr = unsafe { lock.as_mut().get_unchecked_mut() }
                    .as_ref()
                    .getOutput(index);
                let tensor = unsafe { ptr.as_mut() }
                    .ok_or(Error::Runtime("Failed to get output".to_string()))?;

                Ok(Tensor {
                    inner: unsafe { Mutex::new(Pin::new_unchecked(tensor)) },
                })
            }

            fn set_layer_name(&mut self, name: &str) -> Result<()> {
                let name_cstr = std::ffi::CString::new(name)?;
                let lock = self.inner.get_mut()?;
                unsafe {
                    transmute::<&mut Pin<&mut $iface>, &mut Pin<&mut ILayer>>(lock)
                        .as_mut()
                        .setName(name_cstr.as_ptr())
                };
                Ok(())
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
define_network_layer!(SqueezeLayer, ISqueezeLayer);
define_network_layer!(UnsqueezeLayer, IUnsqueezeLayer);
define_network_layer!(NormalizationLayer, INormalizationLayer);

// Those are not actual ILayer in TRT
pub struct Loop<'network> {
    pub(crate) _inner: Mutex<Pin<&'network mut nvinfer1::ILoop>>,
}
pub struct IfConditional<'network> {
    pub(crate) _inner: Mutex<Pin<&'network mut nvinfer1::IIfConditional>>,
}
