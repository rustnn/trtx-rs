//! Network definition for building TensorRT engines
//!
//! Types and implementations. Real/mock impls live in real/ and mock/ folders.

use std::ffi::{CStr, CString};
use std::pin::Pin;
use std::sync::Mutex;
use trtx_sys::TrtLayer;

use trtx_sys::{nvinfer1, LayerType};

/// Panics if the layer or tensor was created from a different network.
#[macro_export]
macro_rules! check_network {
    ($network:ident, $this:ident) => {
        if $network.inner.as_ptr() != $this.network {
            panic!("Layer or tensor was created from different network")
        }
    };
}

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
    pub(crate) inner: Pin<&'network mut nvinfer1::ITensor>,
    pub(crate) network: *const nvinfer1::INetworkDefinition,
}
impl Tensor<'_> {
    pub(crate) unsafe fn new(
        network: *const nvinfer1::INetworkDefinition,
        ptr: *mut nvinfer1::ITensor,
    ) -> Result<Self> {
        unsafe {
            let ptr = ptr.as_mut().ok_or(Error::GetTensorFailed)?;
            Ok(Self {
                inner: Pin::new_unchecked(ptr),
                network,
            })
        }
    }
}

pub struct Layer<'network, Inner: TrtLayer> {
    pub(crate) inner: Pin<&'network mut Inner>,
    pub(crate) network: *const nvinfer1::INetworkDefinition,
}

impl<'network, Inner: TrtLayer> Layer<'network, Inner> {
    pub(crate) fn new(
        network: *const nvinfer1::INetworkDefinition,
        ptr: *mut Inner,
    ) -> Result<Self> {
        unsafe {
            let ptr = ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(Inner::TYPE))?;
            Ok(Self {
                inner: Pin::new_unchecked(ptr),
                network,
            })
        }
    }
    pub const fn layer_type(&self) -> LayerType {
        Inner::TYPE
    }

    /// See [nvinfer1::ILayer::getOutput]
    pub fn get_output(&self, network: &NetworkDefinition, index: i32) -> Result<Tensor<'network>> {
        check_network!(network, self);
        let tensor = self.inner.as_layer().getOutput(index);
        unsafe { Tensor::new(self.network, tensor) }
    }

    /// See [nvinfer1::ILayer::setName]
    pub fn set_name(&mut self, network: &mut NetworkDefinition, name: &str) -> Result<()> {
        check_network!(network, self);
        let name = CString::new(name)?;
        unsafe {
            self.inner
                .as_mut()
                .get_unchecked_mut()
                .as_layer_pin_mut()
                .setName(name.as_ptr())
        };
        Ok(())
    }

    /// See [nvinfer1::ILayer::getName]
    pub fn name(&self, network: &NetworkDefinition) -> String {
        check_network!(network, self);
        let name = self.inner.as_layer().getName();
        // must clone since layer may change name at any time! Cow from to_string_lossy() only
        // possible if name immutable
        if name.is_null() {
            "(unamed)".to_string()
        } else {
            unsafe { CStr::from_ptr(name).to_string_lossy().to_string() }
        }
    }
}

// Type aliases for every layer (Layer<_, I*Layer> where I*Layer: TrtLayer)
pub type ActivationLayer<'layer> = Layer<'layer, nvinfer1::IActivationLayer>;
pub type AssertionLayer<'layer> = Layer<'layer, nvinfer1::IAssertionLayer>;
pub type CastLayer<'layer> = Layer<'layer, nvinfer1::ICastLayer>;
pub type ConcatenationLayer<'layer> = Layer<'layer, nvinfer1::IConcatenationLayer>;
pub type ConstantLayer<'layer> = Layer<'layer, nvinfer1::IConstantLayer>;
pub type ConvolutionLayer<'layer> = Layer<'layer, nvinfer1::IConvolutionLayer>;
pub type CumulativeLayer<'layer> = Layer<'layer, nvinfer1::ICumulativeLayer>;
pub type DeconvolutionLayer<'layer> = Layer<'layer, nvinfer1::IDeconvolutionLayer>;
pub type DequantizeLayer<'layer> = Layer<'layer, nvinfer1::IDequantizeLayer>;
pub type DynamicQuantizeLayer<'layer> = Layer<'layer, nvinfer1::IDynamicQuantizeLayer>;
pub type ElementWiseLayer<'layer> = Layer<'layer, nvinfer1::IElementWiseLayer>;
pub type EinsumLayer<'layer> = Layer<'layer, nvinfer1::IEinsumLayer>;
pub type FillLayer<'layer> = Layer<'layer, nvinfer1::IFillLayer>;
pub type GatherLayer<'layer> = Layer<'layer, nvinfer1::IGatherLayer>;
pub type GridSampleLayer<'layer> = Layer<'layer, nvinfer1::IGridSampleLayer>;
pub type IdentityLayer<'layer> = Layer<'layer, nvinfer1::IIdentityLayer>;
pub type MatrixMultiplyLayer<'layer> = Layer<'layer, nvinfer1::IMatrixMultiplyLayer>;
pub type NMSLayer<'layer> = Layer<'layer, nvinfer1::INMSLayer>;
pub type NonZeroLayer<'layer> = Layer<'layer, nvinfer1::INonZeroLayer>;
pub type NormalizationLayer<'layer> = Layer<'layer, nvinfer1::INormalizationLayer>;
pub type PaddingLayer<'layer> = Layer<'layer, nvinfer1::IPaddingLayer>;
pub type ParametricReLULayer<'layer> = Layer<'layer, nvinfer1::IParametricReLULayer>;
pub type PoolingLayer<'layer> = Layer<'layer, nvinfer1::IPoolingLayer>;
pub type QuantizeLayer<'layer> = Layer<'layer, nvinfer1::IQuantizeLayer>;
pub type RaggedSoftMaxLayer<'layer> = Layer<'layer, nvinfer1::IRaggedSoftMaxLayer>;
pub type ReduceLayer<'layer> = Layer<'layer, nvinfer1::IReduceLayer>;
pub type ResizeLayer<'layer> = Layer<'layer, nvinfer1::IResizeLayer>;
pub type RotaryEmbeddingLayer<'layer> = Layer<'layer, nvinfer1::IRotaryEmbeddingLayer>;
pub type ScaleLayer<'layer> = Layer<'layer, nvinfer1::IScaleLayer>;
pub type ScatterLayer<'layer> = Layer<'layer, nvinfer1::IScatterLayer>;
pub type SelectLayer<'layer> = Layer<'layer, nvinfer1::ISelectLayer>;
pub type ShapeLayer<'layer> = Layer<'layer, nvinfer1::IShapeLayer>;
pub type ShuffleLayer<'layer> = Layer<'layer, nvinfer1::IShuffleLayer>;
pub type SliceLayer<'layer> = Layer<'layer, nvinfer1::ISliceLayer>;
pub type SoftMaxLayer<'layer> = Layer<'layer, nvinfer1::ISoftMaxLayer>;
pub type SqueezeLayer<'layer> = Layer<'layer, nvinfer1::ISqueezeLayer>;
pub type TopKLayer<'layer> = Layer<'layer, nvinfer1::ITopKLayer>;
pub type UnaryLayer<'layer> = Layer<'layer, nvinfer1::IUnaryLayer>;
pub type UnsqueezeLayer<'layer> = Layer<'layer, nvinfer1::IUnsqueezeLayer>;
pub type ReverseSequenceLayer<'layer> = Layer<'layer, nvinfer1::IReverseSequenceLayer>;
pub type KVCacheUpdateLayer<'layer> = Layer<'layer, nvinfer1::IKVCacheUpdateLayer>;
pub type LrnLayer<'layer> = Layer<'layer, nvinfer1::ILRNLayer>;
pub type OneHotLayer<'layer> = Layer<'layer, nvinfer1::IOneHotLayer>;

// Loop and conditional boundary layers (created via Loop / IfConditional / add_attention)
pub type AttentionInputLayer<'layer> = Layer<'layer, nvinfer1::IAttentionInputLayer>;
pub type AttentionOutputLayer<'layer> = Layer<'layer, nvinfer1::IAttentionOutputLayer>;
pub type AttentionBoundaryLayer<'layer> = Layer<'layer, nvinfer1::IAttentionBoundaryLayer>;
pub type LoopBoundaryLayer<'layer> = Layer<'layer, nvinfer1::ILoopBoundaryLayer>;
pub type RecurrenceLayer<'layer> = Layer<'layer, nvinfer1::IRecurrenceLayer>;
pub type LoopOutputLayer<'layer> = Layer<'layer, nvinfer1::ILoopOutputLayer>;
pub type TripLimitLayer<'layer> = Layer<'layer, nvinfer1::ITripLimitLayer>;
pub type IteratorLayer<'layer> = Layer<'layer, nvinfer1::IIteratorLayer>;
pub type ConditionLayer<'layer> = Layer<'layer, nvinfer1::IConditionLayer>;
pub type IfConditionalOutputLayer<'layer> = Layer<'layer, nvinfer1::IIfConditionalOutputLayer>;
pub type IfConditionalInputLayer<'layer> = Layer<'layer, nvinfer1::IIfConditionalInputLayer>;

// Those are not actual ILayer in TRT
pub struct Loop<'network> {
    pub(crate) _inner: Mutex<Pin<&'network mut nvinfer1::ILoop>>,
}
pub struct IfConditional<'network> {
    pub(crate) _inner: Mutex<Pin<&'network mut nvinfer1::IIfConditional>>,
}
