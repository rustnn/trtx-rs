//! Network definition for building TensorRT engines.
//!
//! [`NetworkDefinition`] holds [`trtx_sys::nvinfer1::INetworkDefinition`] (C++ [`nvinfer1::INetworkDefinition`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_network_definition.html)).
//! [`Tensor`] wraps [`trtx_sys::nvinfer1::ITensor`] (C++ [`nvinfer1::ITensor`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_tensor.html)).
//! Layer handles implement [`trtx_sys::AsLayer`] / [`trtx_sys::AsLayerTyped`] over concrete `trtx_sys::nvinfer1::I*Layer` types; C++ index: [annotated](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/annotated.html).

use crate::interfaces::RecordError;
pub use crate::tensor::Tensor;
use cxx::UniquePtr;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::pin::Pin;
use trtx_sys::nvinfer1::{IConcatenationLayer, INetworkDefinition, ITensor};
use trtx_sys::InterpolationMode;
use trtx_sys::{nvinfer1, LayerType, SampleMode, Weights};
use trtx_sys::{AsLayer, AsLayerTyped};
#[cfg(feature = "v_1_4")]
use trtx_sys::{CollectiveOperation, MoEActType, ReduceOperation};
use trtx_sys::{DataType, Dims64, MatrixOperation, ScaleMode, TopKOperation};

/// Panics if the layer or tensor was created from a different network.
macro_rules! check_network {
    ($network:ident, $this:ident) => {
        if $network.inner.as_ptr() != $this.network {
            panic!("Layer or tensor was created from different network")
        }
    };
    ($network:ident, $tensor:expr) => {
        if $network.inner.as_ptr() != $tensor.network {
            panic!("Layer or tensor was created from different network")
        }
    };
}
pub(crate) use check_network;

use crate::error::{Error, OkOrFailedSettingProperty, PropertySetAttempt, Result};
use crate::interfaces::ErrorRecorder;
use log::{debug, trace};

/// Kernel and optional bias weights for convolution and deconvolution layers.
pub struct ConvWeights<'a> {
    pub kernel_weights: &'a [u8],
    pub kernel_dtype: crate::DataType,
    pub bias_weights: Option<&'a [u8]>,
    pub bias_dtype: Option<crate::DataType>,
}

pub struct OwnedWeights {
    pub shape: Vec<i64>,
    pub data_type: DataType,
    pub values: Vec<u8>,
}

pub struct OwnedConvWeights {
    pub kernel: OwnedWeights,
    pub bias: Option<OwnedWeights>,
}

impl OwnedConvWeights {
    pub fn as_weights(&self) -> ConvWeights<'_> {
        ConvWeights {
            kernel_weights: &self.kernel.values,
            kernel_dtype: self.kernel.data_type,
            bias_weights: self.bias.as_ref().map(|b| b.values.as_slice()),
            bias_dtype: self.bias.as_ref().map(|b| b.data_type),
        }
    }
}

/// `Inner` is a concrete [`trtx_sys::nvinfer1`] `I*Layer` (all implement [`trtx_sys::AsLayer`] toward [`trtx_sys::nvinfer1::ILayer`]); C++ base [`nvinfer1::ILayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_layer.html).
pub struct Layer<'network, Inner: AsLayer> {
    pub(crate) inner: Pin<&'network mut Inner>,
    pub(crate) network: *const nvinfer1::INetworkDefinition,
}

impl<'network, Inner: AsLayerTyped> Layer<'network, Inner> {
    /// See [nvinfer1::ILayer::getType] (compile time dispatch)
    pub const fn layer_type(&self) -> LayerType {
        Inner::TYPE
    }

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
}
impl<'network> Layer<'network, nvinfer1::ILayer> {
    /// See [nvinfer1::ILayer::getType] (dynamic dispatch)
    pub fn layer_type_dynamic(&self) -> LayerType {
        self.inner.as_layer().getType().into()
    }

    /// Create a generic ILayer (of unknown type)
    pub(crate) fn new_dyn(
        network: *const nvinfer1::INetworkDefinition,
        ptr: *mut nvinfer1::ILayer,
    ) -> Result<Self> {
        unsafe {
            let ptr = ptr.as_mut().ok_or(Error::GetLayerFailed)?;
            Ok(Self {
                inner: Pin::new_unchecked(ptr),
                network,
            })
        }
    }
}

impl<'network, Inner: AsLayer> Layer<'network, Inner> {
    /// See [nvinfer1::ILayer::setInput]
    pub fn set_input(
        &mut self,
        network: &'_ mut NetworkDefinition,
        index: i32,
        tensor: &'_ Tensor,
    ) -> Result<()> {
        check_network!(network, self);
        debug!(
            "set_input layer={} index={index} tensor={}",
            layer_dbg(network, self),
            tensor_dbg(network, tensor)
        );
        unsafe { self.inner.as_mut().get_unchecked_mut() }
            .as_layer_pin_mut()
            .setInput(index, tensor.pin_mut());
        Ok(())
    }

    /// See [nvinfer1::ILayer::getInput]
    pub fn input(&self, network: &'_ NetworkDefinition, index: i32) -> Result<Tensor<'network>> {
        check_network!(network, self);
        let tensor = self.inner.as_layer().getInput(index);
        unsafe { Tensor::new(self.network, tensor) }
    }

    #[deprecated = "use input instead"]
    pub fn get_input(
        &self,
        network: &'_ NetworkDefinition,
        index: i32,
    ) -> Result<Tensor<'network>> {
        self.input(network, index)
    }

    /// See [nvinfer1::ILayer::getOutput]
    pub fn output(&self, network: &'_ NetworkDefinition, index: i32) -> Result<Tensor<'network>> {
        check_network!(network, self);
        let tensor = self.inner.as_layer().getOutput(index);
        unsafe { Tensor::new(self.network, tensor) }
    }

    #[deprecated = "use output instead"]
    pub fn get_output(
        &self,
        network: &'_ NetworkDefinition,
        index: i32,
    ) -> Result<Tensor<'network>> {
        self.output(network, index)
    }
    /// See [nvinfer1::ILayer::getNbInputs]
    pub fn num_inputs(&self, network: &'_ NetworkDefinition) -> i32 {
        check_network!(network, self);
        self.inner.as_layer().getNbInputs()
    }

    #[deprecated = "use num_inputs instead"]
    pub fn get_num_inputs(&self, network: &'_ NetworkDefinition) -> i32 {
        self.num_inputs(network)
    }

    /// See [nvinfer1::ILayer::getNbOutputs]
    pub fn num_outputs(&self, network: &'_ NetworkDefinition) -> i32 {
        check_network!(network, self);
        self.inner.as_layer().getNbOutputs()
    }

    #[deprecated = "use num_outputs instead"]
    pub fn get_num_outputs(&self, network: &'_ NetworkDefinition) -> i32 {
        self.num_outputs(network)
    }

    /// See [nvinfer1::ILayer::setName]
    pub fn set_name(&mut self, network: &'_ mut NetworkDefinition, name: &str) -> Result<()> {
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

/// [`trtx_sys::nvinfer1::IActivationLayer`] — C++ [`nvinfer1::IActivationLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_activation_layer.html).
pub type ActivationLayer<'layer> = Layer<'layer, nvinfer1::IActivationLayer>;
/// [`trtx_sys::nvinfer1::IAssertionLayer`] — C++ [`nvinfer1::IAssertionLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_assertion_layer.html).
pub type AssertionLayer<'layer> = Layer<'layer, nvinfer1::IAssertionLayer>;
/// [`trtx_sys::nvinfer1::ICastLayer`] — C++ [`nvinfer1::ICastLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_cast_layer.html).
pub type CastLayer<'layer> = Layer<'layer, nvinfer1::ICastLayer>;
/// [`trtx_sys::nvinfer1::IConcatenationLayer`] — C++ [`nvinfer1::IConcatenationLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_concatenation_layer.html).
pub type ConcatenationLayer<'layer> = Layer<'layer, nvinfer1::IConcatenationLayer>;
/// [`trtx_sys::nvinfer1::IConstantLayer`] — C++ [`nvinfer1::IConstantLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_constant_layer.html).
pub type ConstantLayer<'layer> = Layer<'layer, nvinfer1::IConstantLayer>;
/// [`trtx_sys::nvinfer1::IConvolutionLayer`] — C++ [`nvinfer1::IConvolutionLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_convolution_layer.html).
pub type ConvolutionLayer<'layer> = Layer<'layer, nvinfer1::IConvolutionLayer>;
/// [`trtx_sys::nvinfer1::ICumulativeLayer`] — C++ [`nvinfer1::ICumulativeLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_cumulative_layer.html).
pub type CumulativeLayer<'layer> = Layer<'layer, nvinfer1::ICumulativeLayer>;
/// [`trtx_sys::nvinfer1::IDeconvolutionLayer`] — C++ [`nvinfer1::IDeconvolutionLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_deconvolution_layer.html).
pub type DeconvolutionLayer<'layer> = Layer<'layer, nvinfer1::IDeconvolutionLayer>;
/// [`trtx_sys::nvinfer1::IDequantizeLayer`] — C++ [`nvinfer1::IDequantizeLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_dequantize_layer.html).
pub type DequantizeLayer<'layer> = Layer<'layer, nvinfer1::IDequantizeLayer>;
/// [`trtx_sys::nvinfer1::IDynamicQuantizeLayer`] — C++ [`nvinfer1::IDynamicQuantizeLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_dynamic_quantize_layer.html).
pub type DynamicQuantizeLayer<'layer> = Layer<'layer, nvinfer1::IDynamicQuantizeLayer>;
/// [`trtx_sys::nvinfer1::IElementWiseLayer`] — C++ [`nvinfer1::IElementWiseLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_element_wise_layer.html).
pub type ElementWiseLayer<'layer> = Layer<'layer, nvinfer1::IElementWiseLayer>;
/// [`trtx_sys::nvinfer1::IEinsumLayer`] — C++ [`nvinfer1::IEinsumLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_einsum_layer.html).
pub type EinsumLayer<'layer> = Layer<'layer, nvinfer1::IEinsumLayer>;
/// [`trtx_sys::nvinfer1::IFillLayer`] — C++ [`nvinfer1::IFillLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_fill_layer.html).
pub type FillLayer<'layer> = Layer<'layer, nvinfer1::IFillLayer>;
/// [`trtx_sys::nvinfer1::IGatherLayer`] — C++ [`nvinfer1::IGatherLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_gather_layer.html).
pub type GatherLayer<'layer> = Layer<'layer, nvinfer1::IGatherLayer>;
/// [`trtx_sys::nvinfer1::IGridSampleLayer`] — C++ [`nvinfer1::IGridSampleLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_grid_sample_layer.html).
pub type GridSampleLayer<'layer> = Layer<'layer, nvinfer1::IGridSampleLayer>;
/// [`trtx_sys::nvinfer1::IIdentityLayer`] — C++ [`nvinfer1::IIdentityLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_identity_layer.html).
pub type IdentityLayer<'layer> = Layer<'layer, nvinfer1::IIdentityLayer>;
/// [`trtx_sys::nvinfer1::IMatrixMultiplyLayer`] — C++ [`nvinfer1::IMatrixMultiplyLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_matrix_multiply_layer.html).
pub type MatrixMultiplyLayer<'layer> = Layer<'layer, nvinfer1::IMatrixMultiplyLayer>;
/// [`trtx_sys::nvinfer1::INMSLayer`] — C++ [`nvinfer1::INMSLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_n_m_s_layer.html).
pub type NMSLayer<'layer> = Layer<'layer, nvinfer1::INMSLayer>;
/// [`trtx_sys::nvinfer1::INonZeroLayer`] — C++ [`nvinfer1::INonZeroLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_non_zero_layer.html).
pub type NonZeroLayer<'layer> = Layer<'layer, nvinfer1::INonZeroLayer>;
/// [`trtx_sys::nvinfer1::INormalizationLayer`] — C++ [`nvinfer1::INormalizationLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_normalization_layer.html).
pub type NormalizationLayer<'layer> = Layer<'layer, nvinfer1::INormalizationLayer>;
/// [`trtx_sys::nvinfer1::IPaddingLayer`] — C++ [`nvinfer1::IPaddingLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_padding_layer.html).
pub type PaddingLayer<'layer> = Layer<'layer, nvinfer1::IPaddingLayer>;
/// [`trtx_sys::nvinfer1::IParametricReLULayer`] — C++ [`nvinfer1::IParametricReLULayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_parametric_re_l_u_layer.html).
pub type ParametricReLULayer<'layer> = Layer<'layer, nvinfer1::IParametricReLULayer>;
/// [`trtx_sys::nvinfer1::IPoolingLayer`] — C++ [`nvinfer1::IPoolingLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_pooling_layer.html).
pub type PoolingLayer<'layer> = Layer<'layer, nvinfer1::IPoolingLayer>;
/// [`trtx_sys::nvinfer1::IQuantizeLayer`] — C++ [`nvinfer1::IQuantizeLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_quantize_layer.html).
pub type QuantizeLayer<'layer> = Layer<'layer, nvinfer1::IQuantizeLayer>;
/// [`trtx_sys::nvinfer1::IRaggedSoftMaxLayer`] — C++ [`nvinfer1::IRaggedSoftMaxLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_ragged_soft_max_layer.html).
pub type RaggedSoftMaxLayer<'layer> = Layer<'layer, nvinfer1::IRaggedSoftMaxLayer>;
/// [`trtx_sys::nvinfer1::IReduceLayer`] — C++ [`nvinfer1::IReduceLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_reduce_layer.html).
pub type ReduceLayer<'layer> = Layer<'layer, nvinfer1::IReduceLayer>;
/// [`trtx_sys::nvinfer1::IResizeLayer`] — C++ [`nvinfer1::IResizeLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_resize_layer.html).
pub type ResizeLayer<'layer> = Layer<'layer, nvinfer1::IResizeLayer>;
/// [`trtx_sys::nvinfer1::IRotaryEmbeddingLayer`] — C++ [`nvinfer1::IRotaryEmbeddingLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_rotary_embedding_layer.html).
pub type RotaryEmbeddingLayer<'layer> = Layer<'layer, nvinfer1::IRotaryEmbeddingLayer>;
/// [`trtx_sys::nvinfer1::IScaleLayer`] — C++ [`nvinfer1::IScaleLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_scale_layer.html).
pub type ScaleLayer<'layer> = Layer<'layer, nvinfer1::IScaleLayer>;
/// [`trtx_sys::nvinfer1::IScatterLayer`] — C++ [`nvinfer1::IScatterLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_scatter_layer.html).
pub type ScatterLayer<'layer> = Layer<'layer, nvinfer1::IScatterLayer>;
/// [`trtx_sys::nvinfer1::ISelectLayer`] — C++ [`nvinfer1::ISelectLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_select_layer.html).
pub type SelectLayer<'layer> = Layer<'layer, nvinfer1::ISelectLayer>;
/// [`trtx_sys::nvinfer1::IShapeLayer`] — C++ [`nvinfer1::IShapeLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_shape_layer.html).
pub type ShapeLayer<'layer> = Layer<'layer, nvinfer1::IShapeLayer>;
/// [`trtx_sys::nvinfer1::IShuffleLayer`] — C++ [`nvinfer1::IShuffleLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_shuffle_layer.html).
pub type ShuffleLayer<'layer> = Layer<'layer, nvinfer1::IShuffleLayer>;
/// [`trtx_sys::nvinfer1::ISliceLayer`] — C++ [`nvinfer1::ISliceLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_slice_layer.html).
pub type SliceLayer<'layer> = Layer<'layer, nvinfer1::ISliceLayer>;
/// [`trtx_sys::nvinfer1::ISoftMaxLayer`] — C++ [`nvinfer1::ISoftMaxLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_soft_max_layer.html).
pub type SoftMaxLayer<'layer> = Layer<'layer, nvinfer1::ISoftMaxLayer>;
/// [`trtx_sys::nvinfer1::ISqueezeLayer`] — C++ [`nvinfer1::ISqueezeLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_squeeze_layer.html).
pub type SqueezeLayer<'layer> = Layer<'layer, nvinfer1::ISqueezeLayer>;
/// [`trtx_sys::nvinfer1::ITopKLayer`] — C++ [`nvinfer1::ITopKLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_top_k_layer.html).
pub type TopKLayer<'layer> = Layer<'layer, nvinfer1::ITopKLayer>;
/// [`trtx_sys::nvinfer1::IUnaryLayer`] — C++ [`nvinfer1::IUnaryLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_unary_layer.html).
pub type UnaryLayer<'layer> = Layer<'layer, nvinfer1::IUnaryLayer>;
/// [`trtx_sys::nvinfer1::IUnsqueezeLayer`] — C++ [`nvinfer1::IUnsqueezeLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_unsqueeze_layer.html).
pub type UnsqueezeLayer<'layer> = Layer<'layer, nvinfer1::IUnsqueezeLayer>;
/// [`trtx_sys::nvinfer1::IReverseSequenceLayer`] — C++ [`nvinfer1::IReverseSequenceLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_reverse_sequence_layer.html).
pub type ReverseSequenceLayer<'layer> = Layer<'layer, nvinfer1::IReverseSequenceLayer>;
/// [`trtx_sys::nvinfer1::IKVCacheUpdateLayer`] — C++ [`nvinfer1::IKVCacheUpdateLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_k_v_cache_update_layer.html).
pub type KVCacheUpdateLayer<'layer> = Layer<'layer, nvinfer1::IKVCacheUpdateLayer>;
/// [`trtx_sys::nvinfer1::ILRNLayer`] — C++ [`nvinfer1::ILRNLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_l_r_n_layer.html).
pub type LrnLayer<'layer> = Layer<'layer, nvinfer1::ILRNLayer>;
/// [`trtx_sys::nvinfer1::IOneHotLayer`] — C++ [`nvinfer1::IOneHotLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_one_hot_layer.html).
pub type OneHotLayer<'layer> = Layer<'layer, nvinfer1::IOneHotLayer>;
#[cfg(feature = "v_1_4")]
/// [`trtx_sys::nvinfer1::IMoELayer`] — C++ [`nvinfer1::IMoELayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_mo_e_layer.html).
#[cfg(feature = "v_1_4")]
pub type MoELayer<'layer> = Layer<'layer, nvinfer1::IMoELayer>;
#[cfg(feature = "v_1_4")]
/// [`trtx_sys::nvinfer1::IDistCollectiveLayer`] — C++ [`nvinfer1::IDistCollectiveLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_dist_collective_layer.html).
#[cfg(feature = "v_1_4")]
pub type DistCollectiveLayer<'layer> = Layer<'layer, nvinfer1::IDistCollectiveLayer>;

// Loop and conditional boundary layers (created via Loop / IfConditional / add_attention)
/// [`trtx_sys::nvinfer1::IAttentionInputLayer`] — C++ [`nvinfer1::IAttentionInputLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_attention_input_layer.html).
pub type AttentionInputLayer<'layer> = Layer<'layer, nvinfer1::IAttentionInputLayer>;
/// [`trtx_sys::nvinfer1::IAttentionOutputLayer`] — C++ [`nvinfer1::IAttentionOutputLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_attention_output_layer.html).
pub type AttentionOutputLayer<'layer> = Layer<'layer, nvinfer1::IAttentionOutputLayer>;
/// [`trtx_sys::nvinfer1::IAttentionBoundaryLayer`] — C++ [`nvinfer1::IAttentionBoundaryLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_attention_boundary_layer.html).
pub type AttentionBoundaryLayer<'layer> = Layer<'layer, nvinfer1::IAttentionBoundaryLayer>;
/// [`trtx_sys::nvinfer1::ILoopBoundaryLayer`] — C++ [`nvinfer1::ILoopBoundaryLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_loop_boundary_layer.html).
pub type LoopBoundaryLayer<'layer> = Layer<'layer, nvinfer1::ILoopBoundaryLayer>;
/// [`trtx_sys::nvinfer1::IRecurrenceLayer`] — C++ [`nvinfer1::IRecurrenceLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_recurrence_layer.html).
pub type RecurrenceLayer<'layer> = Layer<'layer, nvinfer1::IRecurrenceLayer>;
/// [`trtx_sys::nvinfer1::ILoopOutputLayer`] — C++ [`nvinfer1::ILoopOutputLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_loop_output_layer.html).
pub type LoopOutputLayer<'layer> = Layer<'layer, nvinfer1::ILoopOutputLayer>;
/// [`trtx_sys::nvinfer1::ITripLimitLayer`] — C++ [`nvinfer1::ITripLimitLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_trip_limit_layer.html).
pub type TripLimitLayer<'layer> = Layer<'layer, nvinfer1::ITripLimitLayer>;
/// [`trtx_sys::nvinfer1::IIteratorLayer`] — C++ [`nvinfer1::IIteratorLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_iterator_layer.html).
pub type IteratorLayer<'layer> = Layer<'layer, nvinfer1::IIteratorLayer>;
/// [`trtx_sys::nvinfer1::IConditionLayer`] — C++ [`nvinfer1::IConditionLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_condition_layer.html).
pub type ConditionLayer<'layer> = Layer<'layer, nvinfer1::IConditionLayer>;
/// [`trtx_sys::nvinfer1::IIfConditionalOutputLayer`] — C++ [`nvinfer1::IIfConditionalOutputLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_if_conditional_output_layer.html).
pub type IfConditionalOutputLayer<'layer> = Layer<'layer, nvinfer1::IIfConditionalOutputLayer>;
/// [`trtx_sys::nvinfer1::IIfConditionalInputLayer`] — C++ [`nvinfer1::IIfConditionalInputLayer`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_if_conditional_input_layer.html).
pub type IfConditionalInputLayer<'layer> = Layer<'layer, nvinfer1::IIfConditionalInputLayer>;

pub type DynLayer<'layer> = Layer<'layer, nvinfer1::ILayer>;

/// Attention block (query, key, value → output). Created by [`NetworkDefinition::add_attention`].
/// Input/output layers are managed internally by TensorRT.
pub struct Attention<'network> {
    pub(crate) inner: Pin<&'network mut nvinfer1::IAttention>,
    pub(crate) network: *const nvinfer1::INetworkDefinition,
}

/// Loop construct for recurrent subgraphs. Created by [`NetworkDefinition::add_loop`].
pub struct Loop<'network> {
    pub(crate) inner: Pin<&'network mut nvinfer1::ILoop>,
    pub(crate) network: *const nvinfer1::INetworkDefinition,
}

/// If-conditional construct. Created by [`NetworkDefinition::add_if_conditional`].
pub struct IfConditional<'network> {
    pub(crate) inner: Pin<&'network mut nvinfer1::IIfConditional>,
    pub(crate) network: *const nvinfer1::INetworkDefinition,
}
impl ShuffleLayer<'_> {
    /// See [nvinfer1::IShuffleLayer::setReshapeDimensions]
    pub fn set_reshape_dimensions(
        &mut self,
        network: &mut NetworkDefinition,
        dims: &[i64],
    ) -> Result<()> {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(dims);
        self.inner.as_mut().setReshapeDimensions(&dims_obj);
        Ok(())
    }

    /// See [nvinfer1::IShuffleLayer::setFirstTranspose]
    pub fn set_first_transpose(
        &mut self,
        network: &mut NetworkDefinition,
        order: &[i32],
    ) -> Result<()> {
        check_network!(network, self);
        let mut order_arr = [0i32; 8];
        let n = order.len().min(8);
        order_arr[..n].copy_from_slice(&order[..n]);
        let perm = trtx_sys::nvinfer1::Permutation { order: order_arr };
        self.inner.as_mut().setFirstTranspose(perm);
        Ok(())
    }

    /// See [nvinfer1::IShuffleLayer::setSecondTranspose]
    pub fn set_second_transpose(
        &mut self,
        network: &mut NetworkDefinition,
        order: &[i32],
    ) -> Result<()> {
        check_network!(network, self);
        let mut order_arr = [0i32; 8];
        let n = order.len().min(8);
        order_arr[..n].copy_from_slice(&order[..n]);
        let perm = trtx_sys::nvinfer1::Permutation { order: order_arr };
        self.inner.as_mut().setSecondTranspose(perm);
        Ok(())
    }
}

impl ResizeLayer<'_> {
    // --- IResizeLayer (full) ---
    // `setInput` / `getInput` / `getOutput` / `getNbInputs` / `getNbOutputs` / name accessors:
    // [`Layer::set_input`], [`Layer::get_input`], [`Layer::get_output`], etc.

    /// See [nvinfer1::IResizeLayer::setOutputDimensions]
    pub fn set_output_dimensions(&mut self, network: &mut NetworkDefinition, dims: &[i64]) {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(dims);
        self.inner.as_mut().setOutputDimensions(&dims_obj);
    }

    /// See [nvinfer1::IResizeLayer::getOutputDimensions]
    pub fn output_dimensions(&self, network: &NetworkDefinition) -> Vec<i64> {
        check_network!(network, self);
        let d = self.inner.as_ref().getOutputDimensions();
        d.d[..d.nbDims as usize].to_vec()
    }

    /// See [nvinfer1::IResizeLayer::setScales]
    pub fn set_scales(&mut self, network: &mut NetworkDefinition, scales: &[f32]) {
        check_network!(network, self);
        // SAFETY: `scales.as_ptr()` is valid for `scales.len()` elements. TensorRT copies scale values
        // into the layer definition during this call (same contract as other INetworkDefinition setters).
        unsafe {
            self.inner
                .as_mut()
                .setScales(scales.as_ptr(), scales.len() as i32);
        }
    }

    /// See [nvinfer1::IResizeLayer::getScales]
    ///
    /// Returns `None` if no scales were set or the resize layer is dynamic (per TensorRT).
    pub fn scales(&self, network: &NetworkDefinition) -> Option<Vec<f32>> {
        check_network!(network, self);
        // SAFETY: TensorRT API allows `size == 0` and `scales == nullptr` to return the scale count only.
        let n = unsafe { self.inner.as_ref().getScales(0, std::ptr::null_mut()) };
        if n <= 0 {
            return None;
        }
        let mut buf = vec![0.0_f32; n as usize];
        // SAFETY: `buf` has length `n` (from `getScales` above); pointer is valid for `n` writes.
        let n2 = unsafe { self.inner.as_ref().getScales(n, buf.as_mut_ptr()) };
        if n2 != n {
            return None;
        }
        Some(buf)
    }

    /// See [nvinfer1::IResizeLayer::setResizeMode]
    pub fn set_resize_mode(&mut self, network: &mut NetworkDefinition, mode: trtx_sys::ResizeMode) {
        check_network!(network, self);
        self.inner.as_mut().setResizeMode(mode.into());
    }

    /// See [nvinfer1::IResizeLayer::getResizeMode]
    pub fn resize_mode(&self, network: &NetworkDefinition) -> trtx_sys::ResizeMode {
        check_network!(network, self);
        self.inner.as_ref().getResizeMode().into()
    }

    /// See [nvinfer1::IResizeLayer::setCoordinateTransformation]
    pub fn set_coordinate_transformation(
        &mut self,
        network: &mut NetworkDefinition,
        transform: trtx_sys::ResizeCoordinateTransformation,
    ) {
        check_network!(network, self);
        self.inner
            .as_mut()
            .setCoordinateTransformation(transform.into());
    }

    /// See [nvinfer1::IResizeLayer::getCoordinateTransformation]
    pub fn coordinate_transformation(
        &self,
        network: &NetworkDefinition,
    ) -> trtx_sys::ResizeCoordinateTransformation {
        check_network!(network, self);
        self.inner.as_ref().getCoordinateTransformation().into()
    }

    /// See [nvinfer1::IResizeLayer::setSelectorForSinglePixel]
    pub fn set_selector_for_single_pixel(
        &mut self,
        network: &mut NetworkDefinition,
        selector: trtx_sys::ResizeSelector,
    ) {
        check_network!(network, self);
        self.inner
            .as_mut()
            .setSelectorForSinglePixel(selector.into());
    }

    /// See [nvinfer1::IResizeLayer::getSelectorForSinglePixel]
    pub fn selector_for_single_pixel(
        &self,
        network: &NetworkDefinition,
    ) -> trtx_sys::ResizeSelector {
        check_network!(network, self);
        self.inner.as_ref().getSelectorForSinglePixel().into()
    }

    /// See [nvinfer1::IResizeLayer::setNearestRounding]
    pub fn set_nearest_rounding(
        &mut self,
        network: &mut NetworkDefinition,
        mode: trtx_sys::ResizeRoundMode,
    ) {
        check_network!(network, self);
        self.inner.as_mut().setNearestRounding(mode.into());
    }

    /// See [nvinfer1::IResizeLayer::getNearestRounding]
    pub fn nearest_rounding(&self, network: &NetworkDefinition) -> trtx_sys::ResizeRoundMode {
        check_network!(network, self);
        self.inner.as_ref().getNearestRounding().into()
    }

    /// See [nvinfer1::IResizeLayer::setCubicCoeff]
    pub fn set_cubic_coeff(&mut self, network: &mut NetworkDefinition, a: f32) {
        check_network!(network, self);
        self.inner.as_mut().setCubicCoeff(a);
    }

    /// See [nvinfer1::IResizeLayer::getCubicCoeff]
    pub fn cubic_coeff(&self, network: &NetworkDefinition) -> f32 {
        check_network!(network, self);
        self.inner.as_ref().getCubicCoeff()
    }

    /// See [nvinfer1::IResizeLayer::setExcludeOutside]
    pub fn set_exclude_outside(&mut self, network: &mut NetworkDefinition, exclude: bool) {
        check_network!(network, self);
        self.inner.as_mut().setExcludeOutside(exclude);
    }

    /// See [nvinfer1::IResizeLayer::getExcludeOutside]
    pub fn exclude_outside(&self, network: &NetworkDefinition) -> bool {
        check_network!(network, self);
        self.inner.as_ref().getExcludeOutside()
    }
}

impl GatherLayer<'_> {
    /// See [nvinfer1::IGatherLayer::setMode]
    pub fn set_gather_mode(&mut self, network: &mut NetworkDefinition, mode: trtx_sys::GatherMode) {
        check_network!(network, self);
        self.inner.as_mut().setMode(mode.into());
    }
}

impl<'network> ScatterLayer<'network> {
    /// See [nvinfer1::IScatterLayer::setMode]
    pub fn set_scatter_mode(
        &mut self,
        network: &mut NetworkDefinition,
        mode: trtx_sys::ScatterMode,
    ) {
        check_network!(network, self);
        self.inner.as_mut().setMode(mode.into());
    }
    /// See [nvinfer1::IScatterLayer::setAxis]
    pub fn set_axis(&mut self, network: &'_ mut NetworkDefinition, axis: i32) {
        check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }
}

impl<'network> ConvolutionLayer<'network> {
    /// See [nvinfer1::IConvolutionLayer::setStrideNd]
    pub fn set_stride(&mut self, network: &mut NetworkDefinition, stride: &[i64; 2]) {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(stride);
        self.inner.as_mut().setStrideNd(&dims_obj);
    }
    /// See [nvinfer1::IConvolutionLayer::setPaddingNd]
    pub fn set_padding(&mut self, network: &mut NetworkDefinition, padding: &[i64; 2]) {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(padding);
        self.inner.as_mut().setPaddingNd(&dims_obj);
    }
    /// See [nvinfer1::IConvolutionLayer::setDilationNd]
    pub fn set_dilation(&mut self, network: &mut NetworkDefinition, dilation: &[i64; 2]) {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(dilation);
        self.inner.as_mut().setDilationNd(&dims_obj);
    }
    /// See [nvinfer1::IConvolutionLayer::setNbGroups]
    pub fn set_num_groups(&mut self, network: &mut NetworkDefinition, num_groups: i64) {
        check_network!(network, self);
        self.inner.as_mut().setNbGroups(num_groups);
    }
}

impl<'network> DeconvolutionLayer<'network> {
    /// See [nvinfer1::IDeconvolutionLayer::setStrideNd]
    pub fn set_stride(&mut self, network: &mut NetworkDefinition, stride: &[i64; 2]) -> Result<()> {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(stride);
        self.inner.as_mut().setStrideNd(&dims_obj);
        Ok(())
    }

    /// Set pre-padding (trim this many elements at the start of each spatial dimension of the output).
    /// Pass [pre_h, pre_w] for 2D deconv; TensorRT applies to the spatial dimensions only.
    ///
    /// See [nvinfer1::IDeconvolutionLayer::setPrePadding]
    pub fn set_pre_padding(
        &mut self,
        network: &mut NetworkDefinition,
        padding: &[i64; 2],
    ) -> Result<()> {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(padding);
        self.inner.as_mut().setPrePadding(&dims_obj);
        Ok(())
    }
    /// Set post-padding (trim this many elements at the end of each spatial dimension of the output).
    /// Pass [post_h, post_w] for 2D deconv; TensorRT applies to the spatial dimensions only.
    ///
    /// See [nvinfer1::IDeconvolutionLayer::setPostPadding]
    pub fn set_post_padding(
        &mut self,
        network: &mut NetworkDefinition,
        padding: &[i64; 2],
    ) -> Result<()> {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(padding);
        self.inner.as_mut().setPostPadding(&dims_obj);
        Ok(())
    }

    /// See [nvinfer1::IDeconvolutionLayer::setDilationNd]
    pub fn set_dilation(
        &mut self,
        network: &mut NetworkDefinition,
        dilation: &[i64; 2],
    ) -> Result<()> {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(dilation);
        self.inner.as_mut().setDilationNd(&dims_obj);
        Ok(())
    }

    /// See [nvinfer1::IDeconvolutionLayer::setNbGroups]
    pub fn set_num_groups(
        &mut self,
        network: &mut NetworkDefinition,
        num_groups: i64,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner.as_mut().setNbGroups(num_groups);
        Ok(())
    }
}

impl<'network> PoolingLayer<'network> {
    /// See [nvinfer1::IPoolingLayer::setPoolingType]
    pub fn set_pooling_type(
        &mut self,
        network: &mut NetworkDefinition,
        pooling_type: trtx_sys::PoolingType,
    ) {
        check_network!(network, self);
        self.inner.as_mut().setPoolingType(pooling_type.into());
    }

    /// See [nvinfer1::IPoolingLayer::getPoolingType]
    pub fn pooling_type(&self, network: &NetworkDefinition) -> trtx_sys::PoolingType {
        check_network!(network, self);
        self.inner.as_ref().getPoolingType().into()
    }

    /// See [nvinfer1::IPoolingLayer::setBlendFactor]
    pub fn set_blend_factor(&mut self, network: &mut NetworkDefinition, blend_factor: f32) {
        check_network!(network, self);
        self.inner.as_mut().setBlendFactor(blend_factor);
    }

    /// See [nvinfer1::IPoolingLayer::getBlendFactor]
    pub fn blend_factor(&self, network: &NetworkDefinition) -> f32 {
        check_network!(network, self);
        self.inner.as_ref().getBlendFactor()
    }

    /// See [nvinfer1::IPoolingLayer::setAverageCountExcludesPadding]
    pub fn set_average_count_excludes_padding(
        &mut self,
        network: &mut NetworkDefinition,
        exclusive: bool,
    ) {
        check_network!(network, self);
        self.inner
            .as_mut()
            .setAverageCountExcludesPadding(exclusive);
    }

    /// See [nvinfer1::IPoolingLayer::getAverageCountExcludesPadding]
    pub fn average_count_excludes_padding(&self, network: &NetworkDefinition) -> bool {
        check_network!(network, self);
        self.inner.as_ref().getAverageCountExcludesPadding()
    }

    /// See [nvinfer1::IPoolingLayer::setPrePadding]
    pub fn set_pre_padding(&mut self, network: &mut NetworkDefinition, padding: &[i64]) {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(padding);
        self.inner.as_mut().setPrePadding(&dims_obj);
    }

    /// See [nvinfer1::IPoolingLayer::getPrePadding]
    pub fn pre_padding(&self, network: &NetworkDefinition) -> Vec<i64> {
        check_network!(network, self);
        let d = self.inner.as_ref().getPrePadding();
        d.d[..d.nbDims as usize].to_vec()
    }

    /// See [nvinfer1::IPoolingLayer::setPostPadding]
    pub fn set_post_padding(&mut self, network: &mut NetworkDefinition, padding: &[i64]) {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(padding);
        self.inner.as_mut().setPostPadding(&dims_obj);
    }

    /// See [nvinfer1::IPoolingLayer::getPostPadding]
    pub fn post_padding(&self, network: &NetworkDefinition) -> Vec<i64> {
        check_network!(network, self);
        let d = self.inner.as_ref().getPostPadding();
        d.d[..d.nbDims as usize].to_vec()
    }

    /// See [nvinfer1::IPoolingLayer::setPaddingMode]
    pub fn set_padding_mode(
        &mut self,
        network: &mut NetworkDefinition,
        padding_mode: trtx_sys::PaddingMode,
    ) {
        check_network!(network, self);
        self.inner.as_mut().setPaddingMode(padding_mode.into());
    }

    /// See [nvinfer1::IPoolingLayer::getPaddingMode]
    pub fn padding_mode(&self, network: &NetworkDefinition) -> trtx_sys::PaddingMode {
        check_network!(network, self);
        self.inner.as_ref().getPaddingMode().into()
    }

    /// See [nvinfer1::IPoolingLayer::setWindowSizeNd]
    pub fn set_window_size_nd(&mut self, network: &mut NetworkDefinition, window_size: &[i64]) {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(window_size);
        self.inner.as_mut().setWindowSizeNd(&dims_obj);
    }

    /// See [nvinfer1::IPoolingLayer::getWindowSizeNd]
    pub fn window_size_nd(&self, network: &NetworkDefinition) -> Vec<i64> {
        check_network!(network, self);
        let d = self.inner.as_ref().getWindowSizeNd();
        d.d[..d.nbDims as usize].to_vec()
    }

    /// See [nvinfer1::IPoolingLayer::setStrideNd]
    pub fn set_stride_nd(&mut self, network: &mut NetworkDefinition, stride: &[i64]) {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(stride);
        self.inner.as_mut().setStrideNd(&dims_obj);
    }

    /// See [nvinfer1::IPoolingLayer::getStrideNd]
    pub fn stride_nd(&self, network: &NetworkDefinition) -> Vec<i64> {
        check_network!(network, self);
        let d = self.inner.as_ref().getStrideNd();
        d.d[..d.nbDims as usize].to_vec()
    }

    /// See [nvinfer1::IPoolingLayer::setPaddingNd]
    pub fn set_padding_nd(&mut self, network: &mut NetworkDefinition, padding: &[i64]) {
        check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(padding);
        self.inner.as_mut().setPaddingNd(&dims_obj);
    }

    /// See [nvinfer1::IPoolingLayer::getPaddingNd]
    pub fn padding_nd(&self, network: &NetworkDefinition) -> Vec<i64> {
        check_network!(network, self);
        let d = self.inner.as_ref().getPaddingNd();
        d.d[..d.nbDims as usize].to_vec()
    }
}

impl<'network> DynamicQuantizeLayer<'network> {
    /// See [nvinfer1::IDynamicQuantizeLayer::setAxis]
    pub fn set_axis(&mut self, network: &mut NetworkDefinition, axis: i32) {
        check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }

    /// See [nvinfer1::IDynamicQuantizeLayer::getAxis]
    pub fn axis(&self, network: &NetworkDefinition) -> i32 {
        check_network!(network, self);
        self.inner.getAxis()
    }

    /// See [nvinfer1::IDynamicQuantizeLayer::setBlockShape]
    pub fn set_block_shape(&mut self, network: &mut NetworkDefinition, block_shape: &[i64]) {
        check_network!(network, self);
        let dims = Dims64::from_slice(block_shape);
        self.inner.as_mut().setBlockShape(&dims);
    }

    /// See [nvinfer1::IDynamicQuantizeLayer::getBlockShape]
    pub fn block_shape(&self, network: &NetworkDefinition) -> Dims64 {
        check_network!(network, self);
        self.inner.getBlockShape()
    }

    /// See [nvinfer1::IDynamicQuantizeLayer::setBlockSize]
    pub fn set_block_size(&mut self, network: &mut NetworkDefinition, size: i32) {
        check_network!(network, self);
        self.inner.as_mut().setBlockSize(size);
    }

    /// See [nvinfer1::IDynamicQuantizeLayer::getBlockSize]
    pub fn block_size(&self, network: &NetworkDefinition) -> i32 {
        check_network!(network, self);
        self.inner.getBlockSize()
    }

    /// See [nvinfer1::IDynamicQuantizeLayer::setToType]
    pub fn set_to_type(&mut self, network: &mut NetworkDefinition, to_type: DataType) {
        check_network!(network, self);
        self.inner.as_mut().setToType(to_type.into());
    }

    /// See [nvinfer1::IDynamicQuantizeLayer::getToType]
    pub fn to_type(&self, network: &NetworkDefinition) -> DataType {
        check_network!(network, self);
        self.inner.getToType().into()
    }

    /// See [nvinfer1::IDynamicQuantizeLayer::setScaleType]
    pub fn set_scale_type(&mut self, network: &mut NetworkDefinition, scale_type: DataType) {
        check_network!(network, self);
        self.inner.as_mut().setScaleType(scale_type.into());
    }

    /// See [nvinfer1::IDynamicQuantizeLayer::getScaleType]
    pub fn scale_type(&self, network: &NetworkDefinition) -> DataType {
        check_network!(network, self);
        self.inner.getScaleType().into()
    }
}

impl<'network> QuantizeLayer<'network> {
    /// See [nvinfer1::IQuantizeLayer::setAxis]
    pub fn set_axis(&mut self, network: &mut NetworkDefinition, axis: i32) {
        check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }

    /// See [nvinfer1::IQuantizeLayer::getAxis]
    pub fn axis(&self, network: &NetworkDefinition) -> i32 {
        check_network!(network, self);
        self.inner.getAxis()
    }

    /// See [nvinfer1::IQuantizeLayer::setBlockShape]
    pub fn set_block_shape(
        &mut self,
        network: &mut NetworkDefinition,
        block_shape: &[i64],
    ) -> Result<()> {
        check_network!(network, self);
        let dims = Dims64::from_slice(block_shape);
        self.inner
            .as_mut()
            .setBlockShape(&dims)
            .ok_or_err(PropertySetAttempt::QuantizeLayerBlockShape)
    }

    /// See [nvinfer1::IQuantizeLayer::getBlockShape]
    pub fn block_shape(&self, network: &NetworkDefinition) -> Dims64 {
        check_network!(network, self);
        self.inner.getBlockShape()
    }

    /// See [nvinfer1::IQuantizeLayer::setToType]
    pub fn set_to_type(&mut self, network: &mut NetworkDefinition, to_type: DataType) {
        check_network!(network, self);
        self.inner.as_mut().setToType(to_type.into());
    }

    /// See [nvinfer1::IQuantizeLayer::getToType]
    pub fn to_type(&self, network: &NetworkDefinition) -> DataType {
        check_network!(network, self);
        self.inner.getToType().into()
    }
}

impl<'network> DequantizeLayer<'network> {
    /// See [nvinfer1::IDequantizeLayer::setAxis]
    pub fn set_axis(&mut self, network: &mut NetworkDefinition, axis: i32) {
        check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }

    /// See [nvinfer1::IDequantizeLayer::getAxis]
    pub fn axis(&self, network: &NetworkDefinition) -> i32 {
        check_network!(network, self);
        self.inner.getAxis()
    }

    /// See [nvinfer1::IDequantizeLayer::setBlockShape]
    pub fn set_block_shape(
        &mut self,
        network: &mut NetworkDefinition,
        block_shape: &[i64],
    ) -> Result<()> {
        check_network!(network, self);
        let dims = Dims64::from_slice(block_shape);
        self.inner
            .as_mut()
            .setBlockShape(&dims)
            .ok_or_err(PropertySetAttempt::DequantizeLayerBlockShape)
    }

    /// See [nvinfer1::IDequantizeLayer::getBlockShape]
    pub fn block_shape(&self, network: &NetworkDefinition) -> Dims64 {
        check_network!(network, self);
        self.inner.getBlockShape()
    }

    /// See [nvinfer1::IDequantizeLayer::setToType]
    pub fn set_to_type(&mut self, network: &mut NetworkDefinition, to_type: DataType) {
        check_network!(network, self);
        self.inner.as_mut().setToType(to_type.into());
    }

    /// See [nvinfer1::IDequantizeLayer::getToType]
    pub fn to_type(&self, network: &NetworkDefinition) -> DataType {
        check_network!(network, self);
        self.inner.getToType().into()
    }
}

impl ConcatenationLayer<'_> {
    /// See [nvinfer1::IConcatenationLayer::setAxis]
    pub fn set_axis(&mut self, network: &mut NetworkDefinition, axis: i32) {
        check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }
}
impl NormalizationLayer<'_> {
    /// See [nvinfer1::INormalizationLayer::setEpsilon]
    pub fn set_epsilon(&mut self, network: &mut NetworkDefinition, eps: f32) {
        check_network!(network, self);
        self.inner.as_mut().setEpsilon(eps);
    }
    /// See [nvinfer1::INormalizationLayer::getEpsilon]
    pub fn epsilon(&self, network: &NetworkDefinition) -> f32 {
        check_network!(network, self);
        self.inner.as_ref().getEpsilon()
    }

    #[deprecated = "use epsilon instead"]
    pub fn get_epsilon(&self, network: &NetworkDefinition) -> f32 {
        self.epsilon(network)
    }
    /// See [nvinfer1::INormalizationLayer::setAxes]
    pub fn set_axes(&mut self, network: &mut NetworkDefinition, axes: crate::Axes) {
        check_network!(network, self);
        self.inner.as_mut().setAxes(axes.to_bits());
    }
    /// See [nvinfer1::INormalizationLayer::getAxes]
    pub fn axes(&self, network: &NetworkDefinition) -> crate::Axes {
        check_network!(network, self);
        crate::Axes::from_bits(self.inner.as_ref().getAxes())
    }

    #[deprecated = "use axes instead"]
    pub fn get_axes(&self, network: &NetworkDefinition) -> crate::Axes {
        self.axes(network)
    }
    /// See [nvinfer1::INormalizationLayer::setNbGroups]
    pub fn set_num_groups(&mut self, network: &mut NetworkDefinition, groups: i64) {
        check_network!(network, self);
        self.inner.as_mut().setNbGroups(groups);
    }
    /// See [nvinfer1::INormalizationLayer::getNbGroups]
    pub fn num_groups(&self, network: &NetworkDefinition) -> i64 {
        check_network!(network, self);
        self.inner.as_ref().getNbGroups()
    }

    #[deprecated = "use num_groups instead"]
    pub fn get_num_groups(&self, network: &NetworkDefinition) -> i64 {
        self.num_groups(network)
    }

    // is removed because deprecated in TRT
    //pub fn set_compute_precision(&mut self, network: &mut NetworkDefinition, data_type: DataType) {
    //pub fn get_compute_precision(&self, network: &NetworkDefinition) -> DataType {

    pub fn is_v2(&self, network: &NetworkDefinition) -> bool {
        check_network!(network, self);
        self.inner.as_ref().isV2()
    }
}

#[cfg(feature = "v_1_4")]
impl MoELayer<'_> {
    /// See [`trtx_sys::nvinfer1::IMoELayer::setGatedWeights`].
    pub fn set_gated_weights(
        &mut self,
        network: &mut NetworkDefinition,
        fc_gate_weights: &Tensor,
        fc_up_weights: &Tensor,
        fc_down_weights: &Tensor,
        activation_type: MoEActType,
    ) -> Result<()> {
        check_network!(network, self);
        check_network!(network, fc_gate_weights);
        check_network!(network, fc_up_weights);
        check_network!(network, fc_down_weights);
        self.inner.as_mut().setGatedWeights(
            fc_gate_weights.pin_mut(),
            fc_up_weights.pin_mut(),
            fc_down_weights.pin_mut(),
            activation_type.into(),
        );
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setGatedBiases`].
    pub fn set_gated_biases(
        &mut self,
        network: &mut NetworkDefinition,
        fc_gate_biases: &Tensor,
        fc_up_biases: &Tensor,
        fc_down_biases: &Tensor,
    ) -> Result<()> {
        check_network!(network, self);
        check_network!(network, fc_gate_biases);
        check_network!(network, fc_up_biases);
        check_network!(network, fc_down_biases);
        self.inner.as_mut().setGatedBiases(
            fc_gate_biases.pin_mut(),
            fc_up_biases.pin_mut(),
            fc_down_biases.pin_mut(),
        );
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setActivationType`].
    pub fn set_activation_type(
        &mut self,
        network: &mut NetworkDefinition,
        activation_type: MoEActType,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner
            .as_mut()
            .setActivationType(activation_type.into());
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::getActivationType`].
    pub fn activation_type(&self, network: &NetworkDefinition) -> MoEActType {
        check_network!(network, self);
        self.inner.as_ref().getActivationType().into()
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setQuantizationStatic`].
    pub fn set_quantization_static(
        &mut self,
        network: &mut NetworkDefinition,
        fc_down_activation_scale: &Tensor,
        data_type: DataType,
    ) -> Result<()> {
        check_network!(network, self);
        check_network!(network, fc_down_activation_scale);
        self.inner
            .as_mut()
            .setQuantizationStatic(fc_down_activation_scale.pin_mut(), data_type.into());
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setQuantizationDynamicDblQ`].
    pub fn set_quantization_dynamic_dbl_q(
        &mut self,
        network: &mut NetworkDefinition,
        fc_down_activation_dbl_q_scale: &Tensor,
        data_type: DataType,
        block_shape: &[i64],
        dyn_q_output_scale_type: DataType,
    ) -> Result<()> {
        check_network!(network, self);
        check_network!(network, fc_down_activation_dbl_q_scale);
        let block = trtx_sys::Dims::from_slice(block_shape);
        self.inner.as_mut().setQuantizationDynamicDblQ(
            fc_down_activation_dbl_q_scale.pin_mut(),
            data_type.into(),
            &block,
            dyn_q_output_scale_type.into(),
        );
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setQuantizationToType`].
    pub fn set_quantization_to_type(
        &mut self,
        network: &mut NetworkDefinition,
        type_: DataType,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner.as_mut().setQuantizationToType(type_.into());
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::getQuantizationToType`].
    pub fn quantization_to_type(&self, network: &NetworkDefinition) -> DataType {
        check_network!(network, self);
        self.inner.as_ref().getQuantizationToType().into()
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setQuantizationBlockShape`].
    pub fn set_quantization_block_shape(
        &mut self,
        network: &mut NetworkDefinition,
        block_shape: &[i64],
    ) -> Result<()> {
        check_network!(network, self);
        let block = trtx_sys::Dims::from_slice(block_shape);
        self.inner.as_mut().setQuantizationBlockShape(&block);
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::getQuantizationBlockShape`].
    pub fn quantization_block_shape(&self, network: &NetworkDefinition) -> Vec<i64> {
        check_network!(network, self);
        let d = self.inner.as_ref().getQuantizationBlockShape();
        d.d[..d.nbDims as usize].to_vec()
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setDynQOutputScaleType`].
    pub fn set_dyn_q_output_scale_type(
        &mut self,
        network: &mut NetworkDefinition,
        type_: DataType,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner.as_mut().setDynQOutputScaleType(type_.into());
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::getDynQOutputScaleType`].
    pub fn dyn_q_output_scale_type(&self, network: &NetworkDefinition) -> DataType {
        check_network!(network, self);
        self.inner.as_ref().getDynQOutputScaleType().into()
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setSwigluParams`].
    pub fn set_swiglu_params(
        &mut self,
        network: &mut NetworkDefinition,
        limit: f32,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner.as_mut().setSwigluParams(limit, alpha, beta);
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setSwigluParamLimit`].
    pub fn set_swiglu_param_limit(
        &mut self,
        network: &mut NetworkDefinition,
        limit: f32,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner.as_mut().setSwigluParamLimit(limit);
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::getSwigluParamLimit`].
    pub fn swiglu_param_limit(&self, network: &NetworkDefinition) -> f32 {
        check_network!(network, self);
        self.inner.as_ref().getSwigluParamLimit()
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setSwigluParamAlpha`].
    pub fn set_swiglu_param_alpha(
        &mut self,
        network: &mut NetworkDefinition,
        alpha: f32,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner.as_mut().setSwigluParamAlpha(alpha);
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::getSwigluParamAlpha`].
    pub fn swiglu_param_alpha(&self, network: &NetworkDefinition) -> f32 {
        check_network!(network, self);
        self.inner.as_ref().getSwigluParamAlpha()
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::setSwigluParamBeta`].
    pub fn set_swiglu_param_beta(
        &mut self,
        network: &mut NetworkDefinition,
        beta: f32,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner.as_mut().setSwigluParamBeta(beta);
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::IMoELayer::getSwigluParamBeta`].
    pub fn swiglu_param_beta(&self, network: &NetworkDefinition) -> f32 {
        check_network!(network, self);
        self.inner.as_ref().getSwigluParamBeta()
    }
}

#[cfg(feature = "v_1_4")]
/// `IDistCollectiveLayer` adds no methods beyond [`ILayer`](trtx_sys::nvinfer1::ILayer); use [`Layer`] helpers.
impl DistCollectiveLayer<'_> {}

/// [`trtx_sys::nvinfer1::INetworkDefinition`] — C++ [`nvinfer1::INetworkDefinition`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_network_definition.html).
pub struct NetworkDefinition<'builder> {
    //pub(crate) inner: Mutex<Pin<&'builder mut INetworkDefinition>>,
    pub(crate) inner: UniquePtr<INetworkDefinition>,
    //error_recorder: Option<Rc<RefCell<ErrorRecorder>>>,
    _builder: PhantomData<&'builder trtx_sys::nvinfer1::IBuilder>,
    small_copied_weights: Vec<Vec<u8>>, // for convenience we hold pointers to scalars here
    error_recorder: Option<Pin<Box<ErrorRecorder>>>,
}

fn tensor_dbg(network: &NetworkDefinition<'_>, tensor: &Tensor<'_>) -> String {
    tensor
        .name(network)
        .unwrap_or_else(|_| "(unnamed)".to_string())
}
fn layer_dbg<Inner: AsLayer>(network: &NetworkDefinition<'_>, layer: &Layer<'_, Inner>) -> String {
    layer.name(network)
}

impl<'network> NetworkDefinition<'network> {
    pub(crate) fn from_ptr(ptr: *mut INetworkDefinition) -> Self {
        Self {
            inner: unsafe { UniquePtr::from_raw(ptr) },
            error_recorder: None,
            _builder: Default::default(),
            small_copied_weights: Default::default(),
        }
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addInput`].
    pub fn add_input(
        &mut self,
        name: &str,
        data_type: trtx_sys::DataType,
        dims: &[i64],
    ) -> Result<Tensor<'network>> {
        debug!("add_input name={name:?} data_type={data_type:?} dims={dims:?}");
        let name_cstr = std::ffi::CString::new(name)?;
        let dims_struct = trtx_sys::Dims::from_slice(dims);
        let tensor_ptr = unsafe {
            self.inner
                .pin_mut()
                .addInput(name_cstr.as_ptr(), data_type.into(), &dims_struct)
        };
        unsafe { Tensor::new(self.inner.as_ptr(), tensor_ptr) }
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::markOutput`].
    pub fn mark_output(&mut self, tensor: &'_ Tensor) {
        check_network!(self, tensor);
        debug!("mark_input tensor={}", tensor_dbg(self, tensor));
        self.inner.pin_mut().markOutput(tensor.pin_mut());
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::markDebug`].
    /// Mark a tensor for debugging; at runtime use [`crate::ExecutionContext`] ([`trtx_sys::nvinfer1::IExecutionContext`]) — C++ [`setDebugListener`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_execution_context.html).
    pub fn mark_tensor_debug(&mut self, tensor: &'_ Tensor) -> Result<()> {
        check_network!(self, tensor);
        let success = self.inner.pin_mut().markDebug(tensor.pin_mut());
        if success {
            Ok(())
        } else {
            Err(Error::Runtime("markDebug failed".to_string()))
        }
    }
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::isDebugTensor`].
    /// See [`trtx_sys::nvinfer1::IExecutionContext`] debug APIs (C++ [docs](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_execution_context.html)).
    pub fn is_debug_tensor(&self, tensor: &'_ Tensor) -> bool {
        check_network!(self, tensor);
        self.inner.isDebugTensor(tensor.as_ref())
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getNbInputs`].
    pub fn nb_inputs(&self) -> i32 {
        self.inner.getNbInputs()
    }

    #[deprecated = "use nb_inputs instead"]
    pub fn get_nb_inputs(&self) -> i32 {
        self.nb_inputs()
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getNbOutputs`].
    pub fn nb_outputs(&self) -> i32 {
        self.inner.getNbOutputs()
    }

    #[deprecated = "use nb_outputs instead"]
    pub fn get_nb_outputs(&self) -> i32 {
        self.nb_outputs()
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getInput`].
    pub fn input(&self, index: i32) -> Result<Tensor<'network>> {
        let tensor_ptr = self.inner.getInput(index);
        if tensor_ptr.is_null() {
            return Err(Error::Runtime(format!(
                "Failed to get input at index {}",
                index
            )));
        }
        unsafe { Tensor::new(self.inner.as_ptr(), tensor_ptr) }
    }

    #[deprecated = "use input instead"]
    pub fn get_input(&self, index: i32) -> Result<Tensor<'network>> {
        self.input(index)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getOutput`].
    pub fn output(&self, index: i32) -> Result<Tensor<'network>> {
        let tensor_ptr = self.inner.getOutput(index);
        if tensor_ptr.is_null() {
            return Err(Error::Runtime(format!(
                "Failed to get output at index {}",
                index
            )));
        }
        unsafe { Tensor::new(self.inner.as_ptr(), tensor_ptr) }
    }

    #[deprecated = "use output instead"]
    pub fn get_output(&self, index: i32) -> Result<Tensor<'network>> {
        self.output(index)
    }

    /// Returns an iterator over the network's input tensors.
    pub fn inputs(&self) -> NetworkInputIter<'_, 'network> {
        NetworkInputIter {
            network: self,
            index: 0,
            count: self.nb_inputs(),
        }
    }

    /// Returns an iterator over the network's output tensors.
    pub fn outputs(&self) -> NetworkOutputIter<'_, 'network> {
        NetworkOutputIter {
            network: self,
            index: 0,
            count: self.nb_outputs(),
        }
    }

    /// Number of layers in the network (for introspection/dumping).
    /// See [INetworkDefinition::getNbLayers]
    pub fn nb_layers(&self) -> i32 {
        self.inner.getNbLayers()
    }

    #[deprecated = "use nb_layers instead"]
    pub fn get_nb_layers(&self) -> i32 {
        self.nb_layers()
    }

    pub fn layer(&self, layer_index: i32) -> Result<DynLayer<'network>> {
        let layer_ptr = self.inner.getLayer(layer_index);
        DynLayer::new_dyn(self.inner.as_ptr(), layer_ptr)
    }

    #[deprecated = "use layer instead"]
    pub fn get_layer(&self, layer_index: i32) -> Result<DynLayer<'network>> {
        self.layer(layer_index)
    }

    /// Layer name at index (for introspection/dumping). Returns "(Unnamed)" if null.
    pub fn layer_name(&self, layer_index: i32) -> Result<String> {
        Ok(self.layer(layer_index)?.name(self))
    }

    #[deprecated = "use layer_name instead"]
    pub fn get_layer_name(&self, layer_index: i32) -> Result<String> {
        self.layer_name(layer_index)
    }

    /// Layer type enum value at index (for introspection/dumping). See TensorRT LayerType.
    pub fn layer_type(&self, layer_index: i32) -> Result<LayerType> {
        Ok(self.layer(layer_index)?.layer_type_dynamic())
    }

    #[deprecated = "use layer_type instead"]
    pub fn get_layer_type(&self, layer_index: i32) -> Result<LayerType> {
        self.layer_type(layer_index)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addActivation`].
    pub fn add_activation(
        &mut self,
        input: &'_ Tensor,
        activation_type: trtx_sys::ActivationType,
    ) -> Result<ActivationLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_activation input={} activation_type={activation_type:?}",
            tensor_dbg(self, input)
        );
        let layer_ptr = self
            .inner
            .pin_mut()
            .addActivation(input.pin_mut(), activation_type.into());
        ActivationLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addUnary`].
    pub fn add_unary(
        &mut self,
        input: &'_ Tensor,
        op: trtx_sys::UnaryOperation,
    ) -> Result<UnaryLayer<'network>> {
        check_network!(self, input);
        debug!("add_unary input={} op={op:?}", tensor_dbg(self, input));
        let layer_ptr = self.inner.pin_mut().addUnary(input.pin_mut(), op.into());
        UnaryLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addIdentity`].
    pub fn add_identity(&mut self, input: &'_ Tensor) -> Result<IdentityLayer<'network>> {
        check_network!(self, input);
        debug!("add_identity input={}", tensor_dbg(self, input));
        let layer_ptr = self.inner.pin_mut().addIdentity(input.pin_mut());

        IdentityLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addCast`].
    pub fn add_cast(
        &mut self,
        input: &'_ Tensor,
        to_type: trtx_sys::DataType,
    ) -> Result<CastLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_cast input={} to_type={to_type:?}",
            tensor_dbg(self, input)
        );
        let layer_ptr = self
            .inner
            .pin_mut()
            .addCast(input.pin_mut(), to_type.into());
        CastLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addElementWise`].
    pub fn add_elementwise(
        &mut self,
        input1: &'_ Tensor,
        input2: &'_ Tensor,
        op: trtx_sys::ElementWiseOperation,
    ) -> Result<ElementWiseLayer<'network>> {
        check_network!(self, input1);
        check_network!(self, input2);
        debug!(
            "add_elementwise input1={} input2={} op={op:?}",
            tensor_dbg(self, input1),
            tensor_dbg(self, input2)
        );
        let layer_ptr =
            self.inner
                .pin_mut()
                .addElementWise(input1.pin_mut(), input2.pin_mut(), op.into());
        ElementWiseLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addPoolingNd`].
    pub fn add_pooling(
        &'_ mut self,
        input: &'_ Tensor,
        pooling_type: trtx_sys::PoolingType,
        window_size: &[i64; 2],
    ) -> Result<PoolingLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_pooling input={} pooling_type={pooling_type:?} window_size={window_size:?}",
            tensor_dbg(self, input)
        );
        let window_dims = trtx_sys::Dims::new_2d(window_size[0], window_size[1]);
        let layer_ptr =
            self.inner
                .pin_mut()
                .addPoolingNd(input.pin_mut(), pooling_type.into(), &window_dims);
        PoolingLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addShuffle`].
    pub fn add_shuffle(&'_ mut self, input: &'_ Tensor) -> Result<ShuffleLayer<'network>> {
        check_network!(self, input);
        debug!("add_shuffle input={}", tensor_dbg(self, input));
        let layer_ptr = self.inner.pin_mut().addShuffle(input.pin_mut());
        ShuffleLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addMatrixMultiply`].
    pub fn add_matrix_multiply(
        &'_ mut self,
        input0: &'_ Tensor,
        op0: MatrixOperation,
        input1: &'_ Tensor,
        op1: MatrixOperation,
    ) -> Result<MatrixMultiplyLayer<'network>> {
        check_network!(self, input0);
        check_network!(self, input1);
        debug!(
            "add_matrix_multiply input0={} op0={op0:?} input1={} op1={op1:?}",
            tensor_dbg(self, input0),
            tensor_dbg(self, input1)
        );
        let layer_ptr = self.inner.pin_mut().addMatrixMultiply(
            input0.pin_mut(),
            op0.into(),
            input1.pin_mut(),
            op1.into(),
        );
        MatrixMultiplyLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// Same as [`Self::add_convolution`] but allows to set weights later (e.g. dynamic kernel and bias)
    ///
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConvolutionNd`].
    pub fn add_convolution_deferrred_weights(
        &'_ mut self,
        input: &'_ Tensor,
        nb_output_maps: i32,
        kernel_size: &[i32; 2],
    ) -> Result<ConvolutionLayer<'network>> {
        debug!(
            "add_convolution_deferrred_weights input={} nb_output_maps={nb_output_maps} kernel_size={kernel_size:?}",
            tensor_dbg(self, input)
        );
        let kernel_dims = trtx_sys::Dims::new_2d(kernel_size[0] as i64, kernel_size[1] as i64);
        let layer_ptr = self.inner.pin_mut().addConvolutionNd(
            input.pin_mut(),
            nb_output_maps as i64,
            &kernel_dims,
            Weights {
                type_: nvinfer1::DataType::kFLOAT,
                values: std::ptr::null(),
                count: 0,
            },
            Weights {
                type_: nvinfer1::DataType::kFLOAT,
                values: std::ptr::null(),
                count: 0,
            },
        );
        ConvolutionLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// Same as [`Self::add_convolution`] but takes ownership of weights
    ///
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConvolutionNd`].
    pub fn add_convolution_owned_weights(
        &'_ mut self,
        input: &'_ Tensor,
        nb_output_maps: i32,
        kernel_size: &[i32; 2],
        weights: OwnedConvWeights,
    ) -> Result<ConvolutionLayer<'network>> {
        debug!(
            "add_convolution_owned_weights input={} nb_output_maps={nb_output_maps} kernel_size={kernel_size:?}",
            tensor_dbg(self, input)
        );
        let mut layer =
            self.add_convolution_deferrred_weights(input, nb_output_maps, kernel_size)?;
        let kernel = self
            .add_constant_owned(
                &weights.kernel.shape,
                weights.kernel.values,
                weights.kernel.data_type,
            )?
            .output(self, 0)?;
        layer.set_input(self, 1, &kernel)?;

        if let Some(bias) = weights.bias {
            let bias = self
                .add_constant_owned(&bias.shape, bias.values, bias.data_type)?
                .output(self, 0)?;
            layer.set_input(self, 2, &bias)?;
        }

        Ok(layer)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConvolutionNd`].
    pub fn add_convolution(
        &'_ mut self,
        input: &'_ Tensor,
        nb_output_maps: i32,
        kernel_size: &[i32; 2],
        weights: &ConvWeights<'network>,
    ) -> Result<ConvolutionLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_convolution input={} nb_output_maps={nb_output_maps} kernel_size={kernel_size:?}",
            tensor_dbg(self, input)
        );
        let kernel_dtype = weights.kernel_dtype;
        let kernel_weights = weights.kernel_weights;
        let bias_weights = weights.bias_weights;
        let bias_dtype = weights.bias_dtype;
        let kernel_bpe = kernel_dtype.size_bits() / 8;
        let weight_count = (kernel_weights.len() / kernel_bpe) as i64;
        let bias_dtype_val = bias_dtype.unwrap_or(kernel_dtype);
        let bias_bpe = bias_dtype_val.size_bits() / 8;
        let bias_count = bias_weights
            .map(|b| (b.len() / bias_bpe) as i64)
            .unwrap_or(0);
        let kernel_ptr = if weight_count > 0 {
            kernel_weights.as_ptr() as *const std::ffi::c_void
        } else {
            std::ptr::null()
        };
        let bias_ptr = if bias_count > 0 {
            bias_weights
                .map(|b| b.as_ptr() as *const std::ffi::c_void)
                .unwrap_or(std::ptr::null())
        } else {
            std::ptr::null()
        };
        let kernel_dims = trtx_sys::Dims::new_2d(kernel_size[0] as i64, kernel_size[1] as i64);
        let kernel_w = trtx_sys::nvinfer1::Weights::new_with_type(
            kernel_dtype.into(),
            kernel_ptr,
            weight_count,
        );
        let bias_w =
            trtx_sys::nvinfer1::Weights::new_with_type(bias_dtype_val.into(), bias_ptr, bias_count);
        let layer_ptr = self.inner.pin_mut().addConvolutionNd(
            input.pin_mut(),
            nb_output_maps as i64,
            &kernel_dims,
            kernel_w,
            bias_w,
        );
        ConvolutionLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// Add a 2D deconvolution layer. Same input semantics as convolution: input 0 = activation,
    /// input 1 = kernel tensor (use set_input(1, tensor) when kernel_weights is empty),
    /// input 2 = bias tensor (use set_input(2, tensor) when bias_weights is None/empty).
    ///
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addDeconvolutionNd`].
    pub fn add_deconvolution(
        &mut self,
        input: &'_ Tensor,
        nb_output_maps: i64,
        kernel_size: &[i64; 2],
        weights: &ConvWeights<'network>,
    ) -> Result<DeconvolutionLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_deconvolution input={} nb_output_maps={nb_output_maps} kernel_size={kernel_size:?}",
            tensor_dbg(self, input)
        );
        let kernel_dtype = weights.kernel_dtype;
        let kernel_weights = weights.kernel_weights;
        let bias_weights = weights.bias_weights;
        let bias_dtype = weights.bias_dtype;
        let kernel_bpe = kernel_dtype.size_bits() / 8;
        let weight_count = (kernel_weights.len() / kernel_bpe) as i64;
        let bias_dtype_val = bias_dtype.unwrap_or(kernel_dtype);
        let bias_bpe = bias_dtype_val.size_bits() / 8;
        let bias_count = bias_weights
            .map(|b| (b.len() / bias_bpe) as i64)
            .unwrap_or(0);
        let kernel_ptr = if weight_count > 0 {
            kernel_weights.as_ptr() as *const std::ffi::c_void
        } else {
            std::ptr::null()
        };
        let bias_ptr = if bias_count > 0 {
            bias_weights
                .map(|b| b.as_ptr() as *const std::ffi::c_void)
                .unwrap_or(std::ptr::null())
        } else {
            std::ptr::null()
        };
        let kernel_dims = trtx_sys::Dims::new_2d(kernel_size[0], kernel_size[1]);
        let kernel_w = trtx_sys::nvinfer1::Weights::new_with_type(
            kernel_dtype.into(),
            kernel_ptr,
            weight_count,
        );
        let bias_w =
            trtx_sys::nvinfer1::Weights::new_with_type(bias_dtype_val.into(), bias_ptr, bias_count);
        let layer_ptr = self.inner.pin_mut().addDeconvolutionNd(
            input.pin_mut(),
            nb_output_maps,
            kernel_dims,
            kernel_w,
            bias_w,
        );
        DeconvolutionLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// Add a 2D deconvolution layer. Same input semantics as convolution: input 0 = activation,
    /// input 1 = kernel tensor (use set_input(1, tensor) when kernel_weights is empty),
    /// input 2 = bias tensor (use set_input(2, tensor) when bias_weights is None/empty).
    ///
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addDeconvolutionNd`].
    pub fn add_deconvolution_deferred_weights(
        &mut self,
        input: &'_ Tensor,
        nb_output_maps: i64,
        kernel_size: &[i64; 2],
    ) -> Result<DeconvolutionLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_deconvolution_deferred_weights input={} nb_output_maps={nb_output_maps} kernel_size={kernel_size:?}",
            tensor_dbg(self, input)
        );
        let kernel_dims = trtx_sys::Dims::new_2d(kernel_size[0], kernel_size[1]);
        let layer_ptr = self.inner.pin_mut().addDeconvolutionNd(
            input.pin_mut(),
            nb_output_maps,
            kernel_dims,
            Weights {
                type_: nvinfer1::DataType::kFLOAT,
                values: std::ptr::null(),
                count: 0,
            },
            Weights {
                type_: nvinfer1::DataType::kFLOAT,
                values: std::ptr::null(),
                count: 0,
            },
        );
        DeconvolutionLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// Same as [`Self::add_deconvolution`] but takes ownership of weights
    ///
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addDeconvolutionNd`].
    pub fn add_deconvolution_owned_weights(
        &'_ mut self,
        input: &'_ Tensor,
        nb_output_maps: i64,
        kernel_size: &[i64; 2],
        weights: OwnedConvWeights,
    ) -> Result<DeconvolutionLayer<'network>> {
        debug!(
            "add_deconvolution_owned_weights input={} nb_output_maps={nb_output_maps} kernel_size={kernel_size:?}",
            tensor_dbg(self, input)
        );
        let mut layer =
            self.add_deconvolution_deferred_weights(input, nb_output_maps, kernel_size)?;
        let kernel = self
            .add_constant_owned(
                &weights.kernel.shape,
                weights.kernel.values,
                weights.kernel.data_type,
            )?
            .output(self, 0)?;
        layer.set_input(self, 1, &kernel)?;

        if let Some(bias) = weights.bias {
            let bias = self
                .add_constant_owned(&bias.shape, bias.values, bias.data_type)?
                .output(self, 0)?;
            layer.set_input(self, 2, &bias)?;
        }

        Ok(layer)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConcatenation`].
    pub fn add_concatenation(&self, inputs: &[&'_ Tensor]) -> Result<ConcatenationLayer<'network>> {
        for t in inputs.iter() {
            check_network!(self, t);
        }
        let input_names: Vec<String> = inputs.iter().map(|t| tensor_dbg(self, t)).collect();
        debug!("add_concatenation inputs={input_names:?}");
        let mut input_ptrs: Vec<*mut std::ffi::c_void> = inputs
            .iter()
            .map(|t| t.as_mut() as *mut ITensor as *mut _)
            .collect();
        let layer_ptr = unsafe {
            trtx_sys::network_add_concatenation(
                self.inner.as_mut_ptr() as *mut std::ffi::c_void,
                input_ptrs.as_mut_ptr(),
                inputs.len() as i32,
            )
        } as *mut IConcatenationLayer;
        ConcatenationLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConstant`].
    /// Same as [`Self::add_constant`] just copying the provided weights for small weights like scalars
    pub fn add_small_constant_copied(
        &mut self,
        dims: &[i64],
        weights: &[u8],
        data_type: trtx_sys::DataType,
    ) -> Result<ConstantLayer<'network>> {
        trace!(
            "add_small_constant_copied dims={dims:?} data_type={data_type:?} weights_len={}",
            weights.len()
        );
        unsafe { self.add_constant_unsafe(dims, weights, data_type, true) }
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConstant`].
    /// Same as [`Self::add_constant`] but takes ownership of weights
    pub fn add_constant_owned(
        &mut self,
        dims: &[i64],
        weights: Vec<u8>,
        data_type: trtx_sys::DataType,
    ) -> Result<ConstantLayer<'network>> {
        trace!(
            "add_constant_owned dims={dims:?} data_type={data_type:?} weights_len={}",
            weights.len()
        );
        let element_count: i64 = dims.iter().product();
        let expected_bytes = element_count * data_type.size_bits() as i64 / 8;
        if weights.len() as i64 != expected_bytes {
            panic!(
                "Weight size mismatch: expected {expected_bytes} bytes, got {} bytes",
                weights.len()
            );
        }
        let dims_struct = trtx_sys::Dims::from_slice(dims);
        let weights_struct = trtx_sys::nvinfer1::Weights::new_with_type(
            data_type.into(),
            {
                self.small_copied_weights.push(weights);
                self.small_copied_weights
                    .last()
                    .expect("can't be empty. we just pushed")
                    .as_ptr()
            } as *const std::ffi::c_void,
            element_count,
        );
        let layer_ptr = self
            .inner
            .pin_mut()
            .addConstant(&dims_struct, weights_struct);
        ConstantLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    unsafe fn add_constant_unsafe(
        &mut self,
        dims: &[i64],
        weights: &[u8],
        data_type: trtx_sys::DataType,
        copy: bool,
    ) -> Result<ConstantLayer<'network>> {
        let element_count: i64 = dims.iter().product();
        let expected_bytes = element_count * data_type.size_bits() as i64 / 8;
        if weights.len() as i64 != expected_bytes {
            panic!(
                "Weight size mismatch: expected {expected_bytes} bytes, got {} bytes",
                weights.len()
            );
        }
        let dims_struct = trtx_sys::Dims::from_slice(dims);
        let weights_struct = trtx_sys::nvinfer1::Weights::new_with_type(
            data_type.into(),
            if copy {
                self.small_copied_weights.push(weights.to_vec());
                self.small_copied_weights
                    .last()
                    .expect("can't be empty. we just pushed")
                    .as_ptr()
            } else {
                weights.as_ptr()
            } as *const std::ffi::c_void,
            element_count,
        );
        let layer_ptr = self
            .inner
            .pin_mut()
            .addConstant(&dims_struct, weights_struct);
        ConstantLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConstant`].
    pub fn add_constant(
        &mut self,
        dims: &[i64],
        weights: &'network [u8],
        data_type: trtx_sys::DataType,
    ) -> Result<ConstantLayer<'network>> {
        trace!(
            "add_constant dims={dims:?} data_type={data_type:?} weights_len={}",
            weights.len()
        );
        unsafe { self.add_constant_unsafe(dims, weights, data_type, false) }
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addSoftMax`].
    pub fn add_softmax(
        &mut self,
        input: &'_ Tensor,
        axes: crate::Axes,
    ) -> Result<SoftMaxLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_softmax input={} axes={axes:?}",
            tensor_dbg(self, input)
        );
        let layer_ptr = self.inner.pin_mut().addSoftMax(input.pin_mut());
        let mut rtn = SoftMaxLayer::new(self.inner.as_ptr(), layer_ptr)?;
        rtn.inner.as_mut().setAxes(axes.to_bits());
        Ok(rtn)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addScale`].
    pub fn add_scale(
        &mut self,
        input: &'_ Tensor,
        mode: ScaleMode,
        shift: &[u8],
        scale: &[u8],
        power: &[u8],
    ) -> Result<ScaleLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_scale input={} mode={mode:?} shift_len={} scale_len={} power_len={}",
            tensor_dbg(self, input),
            shift.len(),
            scale.len(),
            power.len()
        );
        let weight_count = match mode {
            ScaleMode::kUNIFORM => 1i64,
            ScaleMode::kCHANNEL => {
                let input_dims = input.dimensions(self)?;
                if input_dims.len() >= 4 {
                    input_dims[1]
                } else if !input_dims.is_empty() {
                    input_dims[0]
                } else {
                    1i64
                }
            }
            ScaleMode::kELEMENTWISE => {
                let input_dims = input.dimensions(self)?;
                input_dims.iter().product::<i64>()
            }
        };

        let shift_w = trtx_sys::nvinfer1::Weights::new_float(
            shift.as_ptr() as *const std::ffi::c_void,
            weight_count,
        );
        let scale_w = trtx_sys::nvinfer1::Weights::new_float(
            scale.as_ptr() as *const std::ffi::c_void,
            weight_count,
        );
        let power_w = trtx_sys::nvinfer1::Weights::new_float(
            power.as_ptr() as *const std::ffi::c_void,
            weight_count,
        );
        let layer_ptr =
            self.inner
                .pin_mut()
                .addScale(input.pin_mut(), mode.into(), shift_w, scale_w, power_w);
        ScaleLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addReduce`].
    pub fn add_reduce(
        &mut self,
        input: &'_ Tensor,
        op: trtx_sys::ReduceOperation,
        axes: crate::Axes,
        keep_dims: bool,
    ) -> Result<ReduceLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_reduce input={} op={op:?} axes={axes:?} keep_dims={keep_dims}",
            tensor_dbg(self, input)
        );
        let axes_bits = axes.to_bits();
        let layer_ptr =
            self.inner
                .pin_mut()
                .addReduce(input.pin_mut(), op.into(), axes_bits, keep_dims);
        ReduceLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addCumulative`].
    pub fn add_cumulative(
        &mut self,
        input: &'_ Tensor,
        axis: i32,
        op: trtx_sys::CumulativeOperation,
        exclusive: bool,
        reverse: bool,
    ) -> Result<CumulativeLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_cumulative input={} axis={axis} op={op:?} exclusive={exclusive} reverse={reverse}",
            tensor_dbg(self, input)
        );
        let axis_bytes = axis.to_le_bytes();
        let axis_constant =
            self.add_small_constant_copied(&[], &axis_bytes, trtx_sys::DataType::kINT32)?;
        let axis_tensor = axis_constant.output(self, 0)?;
        self.add_cumulative_with_axis_tensor(input, &axis_tensor, op, exclusive, reverse)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addCumulative`].
    pub fn add_cumulative_with_axis_tensor(
        &mut self,
        input: &'_ Tensor,
        axis_tensor: &'_ Tensor,
        op: trtx_sys::CumulativeOperation,
        exclusive: bool,
        reverse: bool,
    ) -> Result<CumulativeLayer<'network>> {
        check_network!(self, input);
        check_network!(self, axis_tensor);
        debug!(
            "add_cumulative_with_axis_tensor input={} axis_tensor={} op={op:?} exclusive={exclusive} reverse={reverse}",
            tensor_dbg(self, input),
            tensor_dbg(self, axis_tensor)
        );
        let layer_ptr = self.inner.pin_mut().addCumulative(
            input.pin_mut(),
            axis_tensor.pin_mut(),
            op.into(),
            exclusive,
            reverse,
        );
        CumulativeLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addSlice`].
    pub fn add_slice(
        &mut self,
        input: &'_ Tensor,
        start: &[i64],
        size: &[i64],
        stride: &[i64],
    ) -> Result<SliceLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_slice input={} start={start:?} size={size:?} stride={stride:?}",
            tensor_dbg(self, input)
        );
        if start.len() != size.len() || start.len() != stride.len() {
            return Err(Error::Runtime(
                "start, size, and stride must have the same length".to_string(),
            ));
        }
        let start_dims = trtx_sys::Dims::from_slice(start);
        let size_dims = trtx_sys::Dims::from_slice(size);
        let stride_dims = trtx_sys::Dims::from_slice(stride);
        let layer_ptr =
            self.inner
                .pin_mut()
                .addSlice(input.pin_mut(), &start_dims, &size_dims, &stride_dims);
        SliceLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addTopK`].
    pub fn add_topk(
        &mut self,
        input: &'_ Tensor,
        op: TopKOperation,
        k: i32,
        axes: crate::Axes,
    ) -> Result<TopKLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_topk input={} op={op:?} k={k} axes={axes:?}",
            tensor_dbg(self, input)
        );
        let axes_bits = axes.to_bits();
        let layer_ptr = self
            .inner
            .pin_mut()
            .addTopK(input.pin_mut(), op.into(), k, axes_bits);
        TopKLayer::new(self.inner.as_ptr(), layer_ptr)
    }
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addResize`].
    pub fn add_resize(&mut self, input: &'_ Tensor) -> Result<ResizeLayer<'network>> {
        check_network!(self, input);
        debug!("add_resize input={}", tensor_dbg(self, input));
        let layer_ptr = self.inner.pin_mut().addResize(input.pin_mut());
        ResizeLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addGather`].
    pub fn add_gather(
        &'_ mut self,
        data: &'_ Tensor,
        indices: &'_ Tensor,
        axis: i32,
    ) -> Result<GatherLayer<'network>> {
        check_network!(self, data);
        check_network!(self, indices);
        debug!(
            "add_gather data={} indices={} axis={axis}",
            tensor_dbg(self, data),
            tensor_dbg(self, indices)
        );
        let layer_ptr = self
            .inner
            .pin_mut()
            .addGather(data.pin_mut(), indices.pin_mut(), axis);
        GatherLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addScatter`].
    pub fn add_scatter(
        &mut self,
        data: &'_ Tensor,
        indices: &'_ Tensor,
        updates: &'_ Tensor,
        mode: trtx_sys::ScatterMode,
    ) -> Result<ScatterLayer<'network>> {
        check_network!(self, data);
        check_network!(self, indices);
        check_network!(self, updates);
        debug!(
            "add_scatter data={} indices={} updates={} mode={mode:?}",
            tensor_dbg(self, data),
            tensor_dbg(self, indices),
            tensor_dbg(self, updates)
        );
        let layer_ptr = self.inner.pin_mut().addScatter(
            data.pin_mut(),
            indices.pin_mut(),
            updates.pin_mut(),
            mode.into(),
        );
        ScatterLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addQuantize`].
    pub fn add_quantize(
        &'_ mut self,
        input: &'_ Tensor,
        scale: &'_ Tensor,
        output_type: trtx_sys::DataType,
    ) -> Result<QuantizeLayer<'network>> {
        check_network!(self, input);
        check_network!(self, scale);
        debug!(
            "add_quantize input={} scale={} output_type={output_type:?}",
            tensor_dbg(self, input),
            tensor_dbg(self, scale)
        );
        #[cfg(not(feature = "enterprise"))]
        let layer_ptr =
            self.inner
                .pin_mut()
                .addQuantize(input.pin_mut(), scale.pin_mut(), output_type.into());
        #[cfg(feature = "enterprise")]
        let layer_ptr =
            self.inner
                .pin_mut()
                .addQuantize1(input.pin_mut(), scale.pin_mut(), output_type.into());
        QuantizeLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addDequantize`].
    pub fn add_dequantize(
        &mut self,
        input: &'_ Tensor,
        scale: &'_ Tensor,
        output_type: trtx_sys::DataType,
    ) -> Result<DequantizeLayer<'network>> {
        check_network!(self, input);
        check_network!(self, scale);
        debug!(
            "add_dequantize input={} scale={} output_type={output_type:?}",
            tensor_dbg(self, input),
            tensor_dbg(self, scale)
        );
        #[cfg(not(feature = "enterprise"))]
        let layer_ptr = self.inner.pin_mut().addDequantize(
            input.pin_mut(),
            scale.pin_mut(),
            output_type.into(),
        );
        #[cfg(feature = "enterprise")]
        let layer_ptr = self.inner.pin_mut().addDequantize1(
            input.pin_mut(),
            scale.pin_mut(),
            output_type.into(),
        );
        DequantizeLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addSelect`].
    pub fn add_select(
        &mut self,
        condition: &'_ Tensor,
        then_input: &'_ Tensor,
        else_input: &'_ Tensor,
    ) -> Result<SelectLayer<'network>> {
        check_network!(self, condition);
        check_network!(self, then_input);
        check_network!(self, else_input);
        debug!(
            "add_select condition={} then_input={} else_input={}",
            tensor_dbg(self, condition),
            tensor_dbg(self, then_input),
            tensor_dbg(self, else_input)
        );
        let layer_ptr = self.inner.pin_mut().addSelect(
            condition.pin_mut(),
            then_input.pin_mut(),
            else_input.pin_mut(),
        );
        SelectLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addPaddingNd`].
    pub fn add_padding(
        &mut self,
        input: &'_ Tensor,
        pre_padding: &[i64],
        post_padding: &[i64],
    ) -> Result<PaddingLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_padding input={} pre_padding={pre_padding:?} post_padding={post_padding:?}",
            tensor_dbg(self, input)
        );
        if pre_padding.len() != post_padding.len() {
            return Err(Error::Runtime(
                "pre_padding and post_padding must have the same length".to_string(),
            ));
        }
        let pre_dims = trtx_sys::Dims::from_slice(pre_padding);
        let post_dims = trtx_sys::Dims::from_slice(post_padding);
        let layer_ptr = self
            .inner
            .pin_mut()
            .addPaddingNd(input.pin_mut(), &pre_dims, &post_dims);
        PaddingLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addAssertion`].
    pub fn add_assertion(&mut self, condition: &'_ Tensor, message: &str) -> Result<()> {
        check_network!(self, condition);
        debug!(
            "add_assertion condition={} message={message:?}",
            tensor_dbg(self, condition)
        );
        let message_cstr = std::ffi::CString::new(message)?;
        let layer_ptr = unsafe {
            self.inner
                .pin_mut()
                .addAssertion(condition.pin_mut(), message_cstr.as_ptr())
        };
        let _ = AssertionLayer::new(self.inner.as_ptr(), layer_ptr)?;
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addLoop`].
    pub fn add_loop(&mut self) -> Result<Loop<'network>> {
        debug!("add_loop");
        let loop_ptr = self.inner.pin_mut().addLoop();
        let loop_ptr = unsafe { loop_ptr.as_mut() }
            .ok_or_else(|| Error::Runtime("Failed to add loop".to_string()))?;
        Ok(Loop {
            inner: unsafe { Pin::new_unchecked(loop_ptr) },
            network: self.inner.as_ptr(),
        })
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addIfConditional`].
    pub fn add_if_conditional(&mut self) -> Result<IfConditional<'network>> {
        debug!("add_if_conditional");
        let if_ptr = self.inner.pin_mut().addIfConditional();
        let if_ptr = unsafe { if_ptr.as_mut() }
            .ok_or_else(|| Error::Runtime("Failed to add if conditional".to_string()))?;
        Ok(IfConditional {
            inner: unsafe { Pin::new_unchecked(if_ptr) },
            network: self.inner.as_ptr(),
        })
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addAttention`].
    /// Creates an attention block (internally creates [`AttentionInputLayer`] and [`AttentionOutputLayer`]).
    pub fn add_attention(
        &mut self,
        query: &'_ Tensor,
        key: &'_ Tensor,
        value: &'_ Tensor,
        norm_op: trtx_sys::AttentionNormalizationOp,
        causal: bool,
    ) -> Result<Attention<'network>> {
        check_network!(self, query);
        check_network!(self, key);
        check_network!(self, value);
        debug!(
            "add_attention query={} key={} value={} norm_op={norm_op:?} causal={causal}",
            tensor_dbg(self, query),
            tensor_dbg(self, key),
            tensor_dbg(self, value)
        );
        let attn_ptr = self.inner.pin_mut().addAttention(
            query.pin_mut(),
            key.pin_mut(),
            value.pin_mut(),
            norm_op.into(),
            causal,
        );
        let attn = unsafe { attn_ptr.as_mut() }
            .ok_or_else(|| Error::Runtime("Failed to add attention".to_string()))?;
        Ok(Attention {
            inner: unsafe { Pin::new_unchecked(attn) },
            network: self.inner.as_ptr(),
        })
    }

    #[cfg(feature = "v_1_4")]
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addMoE`].
    pub fn add_moe(
        &mut self,
        hidden_states: &Tensor,
        selected_experts_for_tokens: &Tensor,
        scores_for_selected_experts: &Tensor,
    ) -> Result<MoELayer<'network>> {
        check_network!(self, hidden_states);
        check_network!(self, selected_experts_for_tokens);
        check_network!(self, scores_for_selected_experts);
        debug!(
            "add_moe hidden_states={} selected_experts_for_tokens={} scores_for_selected_experts={}",
            tensor_dbg(self, hidden_states),
            tensor_dbg(self, selected_experts_for_tokens),
            tensor_dbg(self, scores_for_selected_experts)
        );
        let layer_ptr = self.inner.pin_mut().addMoE(
            hidden_states.pin_mut(),
            selected_experts_for_tokens.pin_mut(),
            scores_for_selected_experts.pin_mut(),
        );
        MoELayer::new(self.inner.as_ptr(), layer_ptr)
    }

    #[cfg(feature = "v_1_4")]
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addDistCollective`].
    ///
    /// Pass an empty `groups` slice so all ranks participate (`groups == nullptr`, `groupSize == 0` in C++).
    pub fn add_dist_collective(
        &mut self,
        input: &Tensor,
        dist_collective_op: CollectiveOperation,
        reduce_op: ReduceOperation,
        root: i64,
        groups: &[i64],
    ) -> Result<DistCollectiveLayer<'network>> {
        check_network!(self, input);
        debug!(
            "add_dist_collective input={} dist_collective_op={dist_collective_op:?} reduce_op={reduce_op:?} root={root} groups={groups:?}",
            tensor_dbg(self, input)
        );
        let (groups_ptr, group_size) = if groups.is_empty() {
            (std::ptr::null_mut(), 0i64)
        } else {
            (groups.as_ptr() as *mut i64, groups.len() as i64)
        };
        let layer_ptr = unsafe {
            self.inner.pin_mut().addDistCollective(
                input.pin_mut(),
                dist_collective_op.into(),
                reduce_op.into(),
                root,
                groups_ptr,
                group_size,
            )
        };
        DistCollectiveLayer::new(self.inner.as_ptr(), layer_ptr)
    }
}

// --- Network input/output iterators ---

/// Iterator over a [`NetworkDefinition`]'s input tensors. Created by [`NetworkDefinition::inputs`].
pub struct NetworkInputIter<'a, 'network> {
    network: &'a NetworkDefinition<'network>,
    index: i32,
    count: i32,
}

impl<'network> Iterator for NetworkInputIter<'_, 'network> {
    type Item = Tensor<'network>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.count {
            return None;
        }
        let tensor = self.network.input(self.index).expect("valid input index");
        self.index += 1;
        Some(tensor)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.count - self.index).max(0) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NetworkInputIter<'_, '_> {}

/// Iterator over a [`NetworkDefinition`]'s output tensors. Created by [`NetworkDefinition::outputs`].
pub struct NetworkOutputIter<'a, 'network> {
    network: &'a NetworkDefinition<'network>,
    index: i32,
    count: i32,
}

impl<'network> Iterator for NetworkOutputIter<'_, 'network> {
    type Item = Tensor<'network>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.count {
            return None;
        }
        let tensor = self.network.output(self.index).expect("valid output index");
        self.index += 1;
        Some(tensor)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.count - self.index).max(0) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NetworkOutputIter<'_, '_> {}

// --- IAttention ---

impl<'network> Attention<'network> {
    /// See [`trtx_sys::nvinfer1::IAttention::setNormalizationOperation`].
    pub fn set_normalization_operation(
        &mut self,
        network: &mut NetworkDefinition,
        op: trtx_sys::AttentionNormalizationOp,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner
            .as_mut()
            .setNormalizationOperation(op.into())
            .ok_or_err(PropertySetAttempt::AttentionLayerNormalizationOp)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getNormalizationOperation`].
    pub fn normalization_operation(
        &self,
        network: &NetworkDefinition,
    ) -> trtx_sys::AttentionNormalizationOp {
        check_network!(network, self);
        self.inner.getNormalizationOperation().into()
    }

    /// See [`trtx_sys::nvinfer1::IAttention::setMask`].
    pub fn set_mask(&mut self, network: &mut NetworkDefinition, mask: &Tensor) -> Result<()> {
        check_network!(network, self);
        check_network!(network, mask);
        self.inner
            .as_mut()
            .setMask(mask.pin_mut())
            .ok_or_err(PropertySetAttempt::AttentionLayerMask)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getMask`]. Returns [`None`] if no mask is set.
    ///
    /// Note: C++ exposes `getMask` as a non-`const` method; this wrapper takes `&mut self` accordingly.
    pub fn mask(&mut self, network: &mut NetworkDefinition) -> Option<Tensor<'network>> {
        check_network!(network, self);
        let p = self.inner.as_mut().getMask();
        if p.is_null() {
            None
        } else {
            unsafe { Tensor::new(self.network, p).ok() }
        }
    }

    /// See [`trtx_sys::nvinfer1::IAttention::setCausal`].
    pub fn set_causal(&mut self, network: &mut NetworkDefinition, is_causal: bool) -> Result<()> {
        check_network!(network, self);
        self.inner
            .as_mut()
            .setCausal(is_causal)
            .ok_or_err(PropertySetAttempt::AttentionLayerCausal)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getCausal`].
    pub fn causal(&self, network: &NetworkDefinition) -> bool {
        check_network!(network, self);
        self.inner.getCausal()
    }

    /// See [`trtx_sys::nvinfer1::IAttention::setDecomposable`].
    pub fn set_decomposable(
        &mut self,
        network: &mut NetworkDefinition,
        decomposable: bool,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner
            .as_mut()
            .setDecomposable(decomposable)
            .ok_or_err(PropertySetAttempt::AttentionLayerDecomposable)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getDecomposable`].
    pub fn decomposable(&self, network: &NetworkDefinition) -> bool {
        check_network!(network, self);
        self.inner.getDecomposable()
    }

    /// See [`trtx_sys::nvinfer1::IAttention::setInput`].
    pub fn set_input(
        &mut self,
        network: &mut NetworkDefinition,
        index: i32,
        input: &Tensor,
    ) -> Result<()> {
        check_network!(network, self);
        check_network!(network, input);
        self.inner
            .as_mut()
            .setInput(index, input.pin_mut())
            .ok_or_err(PropertySetAttempt::AttentionLayerInput)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getNbInputs`].
    pub fn num_inputs(&self, network: &NetworkDefinition) -> i32 {
        check_network!(network, self);
        self.inner.getNbInputs()
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getInput`].
    pub fn input(&self, network: &NetworkDefinition, index: i32) -> Result<Tensor<'network>> {
        check_network!(network, self);
        let tensor_ptr = self.inner.getInput(index);
        unsafe { Tensor::new(self.network, tensor_ptr) }
    }

    #[deprecated = "use input instead"]
    pub fn get_input(&self, network: &NetworkDefinition, index: i32) -> Result<Tensor<'network>> {
        self.input(network, index)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getNbOutputs`].
    pub fn num_outputs(&self, network: &NetworkDefinition) -> i32 {
        check_network!(network, self);
        self.inner.getNbOutputs()
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getOutput`]. IAttention has one output (index 0).
    pub fn output(&self, network: &NetworkDefinition, index: i32) -> Result<Tensor<'network>> {
        check_network!(network, self);
        let tensor_ptr = self.inner.getOutput(index);
        unsafe { Tensor::new(self.network, tensor_ptr) }
    }

    #[deprecated = "use output instead"]
    pub fn get_output(&self, network: &NetworkDefinition, index: i32) -> Result<Tensor<'network>> {
        self.output(network, index)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::setName`].
    ///
    /// The C++ API requires a null-terminated name of at most 4096 bytes including the terminator.
    pub fn set_name(&mut self, network: &mut NetworkDefinition, name: &str) -> Result<()> {
        check_network!(network, self);
        let name = CString::new(name)?;
        unsafe { self.inner.as_mut().setName(name.as_ptr()) }
            .ok_or_err(PropertySetAttempt::AttentionLayerName)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getName`].
    pub fn name(&self, network: &NetworkDefinition) -> String {
        check_network!(network, self);
        let name = self.inner.getName();
        if name.is_null() {
            "(unamed)".to_string()
        } else {
            unsafe { CStr::from_ptr(name).to_string_lossy().to_string() }
        }
    }

    /// See [`trtx_sys::nvinfer1::IAttention::setNormalizationQuantizeScale`].
    pub fn set_normalization_quantize_scale(
        &mut self,
        network: &mut NetworkDefinition,
        tensor: &Tensor,
    ) -> Result<()> {
        check_network!(network, self);
        check_network!(network, tensor);
        self.inner
            .as_mut()
            .setNormalizationQuantizeScale(tensor.pin_mut())
            .ok_or_err(PropertySetAttempt::AttentionLayerQuantizeScale)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getNormalizationQuantizeScale`].
    pub fn normalization_quantize_scale(
        &self,
        network: &NetworkDefinition,
    ) -> Option<Tensor<'network>> {
        check_network!(network, self);
        let p = self.inner.getNormalizationQuantizeScale();
        if p.is_null() {
            None
        } else {
            unsafe { Tensor::new(self.network, p).ok() }
        }
    }

    /// See [`trtx_sys::nvinfer1::IAttention::setNormalizationQuantizeToType`].
    pub fn set_normalization_quantize_to_type(
        &mut self,
        network: &mut NetworkDefinition,
        type_: DataType,
    ) -> Result<()> {
        check_network!(network, self);
        self.inner
            .as_mut()
            .setNormalizationQuantizeToType(type_.into())
            .ok_or_err(PropertySetAttempt::AttentionLayerQuantizeToType)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getNormalizationQuantizeToType`].
    pub fn normalization_quantize_to_type(&self, network: &NetworkDefinition) -> DataType {
        check_network!(network, self);
        self.inner.getNormalizationQuantizeToType().into()
    }

    #[deprecated = "use normalization_quantize_to_type instead"]
    pub fn get_normalization_quantize_to_type(&self, network: &NetworkDefinition) -> DataType {
        self.normalization_quantize_to_type(network)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::setMetadata`].
    pub fn set_metadata(&mut self, network: &mut NetworkDefinition, metadata: &str) -> Result<()> {
        check_network!(network, self);
        let metadata_cstr = CString::new(metadata)?;
        unsafe { self.inner.as_mut().setMetadata(metadata_cstr.as_ptr()) }
            .ok_or_err(PropertySetAttempt::AttentionLayerMetadata)
    }

    /// See [`trtx_sys::nvinfer1::IAttention::getMetadata`].
    pub fn metadata(&self, network: &NetworkDefinition) -> String {
        check_network!(network, self);
        let p = self.inner.getMetadata();
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p).to_string_lossy().to_string() }
        }
    }

    #[cfg(feature = "v_1_4")]
    /// See [`trtx_sys::nvinfer1::IAttention::setNbRanks`].
    pub fn set_nb_ranks(&mut self, network: &mut NetworkDefinition, nb_ranks: i32) -> Result<()> {
        check_network!(network, self);
        self.inner
            .as_mut()
            .setNbRanks(nb_ranks)
            .ok_or_err(PropertySetAttempt::AttentionLayerNumRanks)
    }

    #[cfg(feature = "v_1_4")]
    /// See [`trtx_sys::nvinfer1::IAttention::getNbRanks`].
    pub fn nb_ranks(&self, network: &NetworkDefinition) -> i32 {
        check_network!(network, self);
        self.inner.getNbRanks()
    }

    #[deprecated = "use nb_ranks instead"]
    pub fn get_nb_ranks(&self, network: &NetworkDefinition) -> i32 {
        self.nb_ranks(network)
    }
}

// --- Loop boundary layers (ILoop::addRecurrence, addTripLimit, addIterator, addLoopOutput) ---

impl<'network> Loop<'network> {
    /// See [`trtx_sys::nvinfer1::ILoop::addRecurrence`].
    pub fn add_recurrence(
        &mut self,
        network: &mut NetworkDefinition,
        initial_value: &'_ Tensor,
    ) -> Result<RecurrenceLayer<'network>> {
        check_network!(network, self);
        check_network!(network, initial_value);
        debug!(
            "Loop::add_recurrence initial_value={}",
            tensor_dbg(network, initial_value)
        );
        let layer_ptr = { self.inner.as_mut().addRecurrence(initial_value.pin_mut()) };
        RecurrenceLayer::new(network.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::ILoop::addTripLimit`].
    pub fn add_trip_limit(
        &mut self,
        network: &mut NetworkDefinition,
        tensor: &'_ Tensor,
        limit: trtx_sys::TripLimit,
    ) -> Result<TripLimitLayer<'network>> {
        check_network!(network, self);
        check_network!(network, tensor);
        debug!(
            "Loop::add_trip_limit tensor={} limit={limit:?}",
            tensor_dbg(network, tensor)
        );
        let layer_ptr = {
            self.inner
                .as_mut()
                .addTripLimit(tensor.pin_mut(), limit.into())
        };
        TripLimitLayer::new(network.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::ILoop::addIterator`].
    pub fn add_iterator(
        &mut self,
        network: &mut NetworkDefinition,
        tensor: &'_ Tensor,
        axis: i32,
        reverse: bool,
    ) -> Result<IteratorLayer<'network>> {
        check_network!(network, self);
        check_network!(network, tensor);
        debug!(
            "Loop::add_iterator tensor={} axis={axis} reverse={reverse}",
            tensor_dbg(network, tensor)
        );
        let layer_ptr = {
            self.inner
                .as_mut()
                .addIterator(tensor.pin_mut(), axis, reverse)
        };
        IteratorLayer::new(network.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::ILoop::addLoopOutput`].
    pub fn add_loop_output(
        &mut self,
        network: &mut NetworkDefinition,
        tensor: &'_ Tensor,
        output_kind: trtx_sys::LoopOutput,
        axis: i32,
    ) -> Result<LoopOutputLayer<'network>> {
        check_network!(network, self);
        check_network!(network, tensor);
        debug!(
            "Loop::add_loop_output tensor={} output_kind={output_kind:?} axis={axis}",
            tensor_dbg(network, tensor)
        );
        let layer_ptr =
            self.inner
                .as_mut()
                .addLoopOutput(tensor.pin_mut(), output_kind.into(), axis);
        LoopOutputLayer::new(network.inner.as_ptr(), layer_ptr)
    }
}

// --- IfConditional boundary layers (IIfConditional::setCondition, addInput, addOutput) ---

impl<'network> IfConditional<'network> {
    /// See [`trtx_sys::nvinfer1::IIfConditional::setCondition`].
    pub fn set_condition(
        &mut self,
        network: &mut NetworkDefinition,
        condition: &'_ Tensor,
    ) -> Result<ConditionLayer<'network>> {
        check_network!(network, self);
        check_network!(network, condition);
        let layer_ptr = self.inner.as_mut().setCondition(condition.pin_mut());
        ConditionLayer::new(network.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::IIfConditional::addInput`].
    pub fn add_input(
        &mut self,
        network: &mut NetworkDefinition,
        input: &'_ Tensor,
    ) -> Result<IfConditionalInputLayer<'network>> {
        check_network!(network, self);
        check_network!(network, input);
        debug!(
            "IfConditional::add_input input={}",
            tensor_dbg(network, input)
        );
        let layer_ptr = self.inner.as_mut().addInput(input.pin_mut());
        IfConditionalInputLayer::new(network.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::IIfConditional::addOutput`].
    pub fn add_output(
        &mut self,
        network: &mut NetworkDefinition,
        true_output: &'_ Tensor,
        false_output: &'_ Tensor,
    ) -> Result<IfConditionalOutputLayer<'network>> {
        check_network!(network, self);
        check_network!(network, true_output);
        check_network!(network, false_output);
        debug!(
            "IfConditional::add_output true_output={} false_output={}",
            tensor_dbg(network, true_output),
            tensor_dbg(network, false_output)
        );
        let layer_ptr = self
            .inner
            .as_mut()
            .addOutput(true_output.pin_mut(), false_output.pin_mut());
        IfConditionalOutputLayer::new(network.inner.as_ptr(), layer_ptr)
    }
}

// --- RecurrenceLayer: set_input(1, tensor) for value from inside loop ---

/// See [`trtx_sys::nvinfer1::IRecurrenceLayer`]. Input 0 = initial value (set at creation); input 1 = value from previous iteration (from inside loop).
impl<'network> RecurrenceLayer<'network> {}

// --- IteratorLayer: set_axis, set_reverse ---

impl IteratorLayer<'_> {
    /// See [`trtx_sys::nvinfer1::IIteratorLayer::setAxis`].
    pub fn set_axis(&mut self, network: &mut NetworkDefinition, axis: i32) {
        check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }
    /// See [`trtx_sys::nvinfer1::IIteratorLayer::setReverse`].
    pub fn set_reverse(&mut self, network: &mut NetworkDefinition, reverse: bool) {
        check_network!(network, self);
        self.inner.as_mut().setReverse(reverse);
    }
}

// --- LoopOutputLayer: get_loop_output, set_axis (for concatenation), set_input for index 1 ---

impl LoopOutputLayer<'_> {
    /// See [`trtx_sys::nvinfer1::ILoopOutputLayer::getLoopOutput`].
    pub fn loop_output(&self, network: &NetworkDefinition) -> trtx_sys::nvinfer1::LoopOutput {
        check_network!(network, self);
        self.inner.as_ref().getLoopOutput()
    }

    #[deprecated = "use loop_output instead"]
    pub fn get_loop_output(&self, network: &NetworkDefinition) -> trtx_sys::nvinfer1::LoopOutput {
        self.loop_output(network)
    }
    /// See [`trtx_sys::nvinfer1::ILoopOutputLayer::setAxis`]. Ignored if output kind is kLAST_VALUE.
    pub fn set_axis(&mut self, network: &mut NetworkDefinition, axis: i32) {
        check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }
}

// --- TripLimitLayer: get_trip_limit (getter only) ---

impl TripLimitLayer<'_> {
    /// See [`trtx_sys::nvinfer1::ITripLimitLayer::getTripLimit`].
    pub fn trip_limit(&self, network: &NetworkDefinition) -> trtx_sys::nvinfer1::TripLimit {
        check_network!(network, self);
        self.inner.as_ref().getTripLimit()
    }

    #[deprecated = "use trip_limit instead"]
    pub fn get_trip_limit(&self, network: &NetworkDefinition) -> trtx_sys::nvinfer1::TripLimit {
        self.trip_limit(network)
    }
}

impl<'builder> NetworkDefinition<'builder> {
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addNormalization`].
    pub fn add_normalization(
        &mut self,
        input: &'_ Tensor,
        scale: &'_ Tensor,
        bias: &'_ Tensor,
        axes_mask: crate::Axes,
    ) -> Result<NormalizationLayer<'builder>> {
        check_network!(self, input);
        check_network!(self, scale);
        check_network!(self, bias);
        debug!(
            "add_normalization input={} scale={} bias={} axes_mask={axes_mask:?}",
            tensor_dbg(self, input),
            tensor_dbg(self, scale),
            tensor_dbg(self, bias)
        );
        let axes_bits = axes_mask.to_bits();
        let ptr = self.inner.pin_mut().addNormalization(
            input.pin_mut(),
            scale.pin_mut(),
            bias.pin_mut(),
            axes_bits,
        );
        NormalizationLayer::new(self.inner.as_ptr(), ptr)
    }
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addNormalizationV2`].
    pub fn add_normalization_v2(
        &mut self,
        input: &'_ Tensor,
        scale: &'_ Tensor,
        bias: &'_ Tensor,
        axes_mask: crate::Axes,
    ) -> Result<NormalizationLayer<'builder>> {
        check_network!(self, input);
        check_network!(self, scale);
        check_network!(self, bias);
        debug!(
            "add_normalization_v2 input={} scale={} bias={} axes_mask={axes_mask:?}",
            tensor_dbg(self, input),
            tensor_dbg(self, scale),
            tensor_dbg(self, bias)
        );
        let axes_bits = axes_mask.to_bits();
        let ptr = self.inner.pin_mut().addNormalizationV2(
            input.pin_mut(),
            scale.pin_mut(),
            bias.pin_mut(),
            axes_bits,
        );
        NormalizationLayer::new(self.inner.as_ptr(), ptr)
    }

    /// See [y nvinfer1::INetworkDefinition::setErrorRecorder]
    ///
    /// The Rust bindings only allow setting the error recorder once
    pub fn set_error_recorder(&mut self, error_recorder: Box<dyn RecordError>) -> Result<()> {
        let error_recorder = ErrorRecorder::new(error_recorder)?;
        if self.error_recorder.is_some() {
            // would need to make sure that we don't destroy a monitor still in use
            // could offer this as an unsafe method for users who only set this when there is no
            // build process active. Or we only accept a ref to progress monitor and force user
            // via lifetimes to keep this alive for builder config lifetime
            panic!("Setting a progress monitor more than once not supported at the moment");
        }
        self.error_recorder = Some(error_recorder);
        let rec = self
            .error_recorder
            .as_mut()
            .unwrap()
            .as_trt_error_recorder();
        #[cfg(not(feature = "mock"))]
        unsafe {
            self.inner.pin_mut().setErrorRecorder(rec)
        };
        Ok(())
    }

    pub fn add_grid_sample(
        &mut self,
        input: &'_ Tensor,
        grid: &'_ Tensor,
    ) -> Result<GridSampleLayer<'builder>> {
        check_network!(self, input);
        check_network!(self, grid);

        let ptr = self
            .inner
            .pin_mut()
            .addGridSample(input.pin_mut(), grid.pin_mut());
        GridSampleLayer::new(self.inner.as_ptr(), ptr)
    }
}

impl<'network> GridSampleLayer<'network> {
    /// See [nvinfer1::IGridSampleLayer::setInterpolationMode]
    pub fn set_interpolation_mode(
        &mut self,
        network: &mut NetworkDefinition,
        mode: InterpolationMode,
    ) {
        check_network!(network, self);
        self.inner.as_mut().setInterpolationMode(mode.into());
    }
    /// See [nvinfer1::IGridSampleLayer::getInterpolationMode]
    pub fn interpolation_mode(&self, network: &NetworkDefinition) -> InterpolationMode {
        check_network!(network, self);
        self.inner.getInterpolationMode().into()
    }
    /// See [nvinfer1::IGridSampleLayer::setSampleMode]
    pub fn set_sample_mode(&mut self, network: &mut NetworkDefinition, mode: SampleMode) {
        check_network!(network, self);
        self.inner.as_mut().setSampleMode(mode.into());
    }
    /// See [nvinfer1::IGridSampleLayer::getSampleMode]
    pub fn sample_mode(&self, network: &NetworkDefinition) -> SampleMode {
        check_network!(network, self);
        self.inner.getSampleMode().into()
    }
    /// See [nvinfer1::IGridSampleLayer::setAlignCorners]
    pub fn set_align_corners(&mut self, network: &mut NetworkDefinition, align_corners: bool) {
        check_network!(network, self);
        self.inner.as_mut().setAlignCorners(align_corners);
    }
    /// See [nvinfer1::IGridSampleLayer::getAlignCorners]
    pub fn align_corners(&self, network: &NetworkDefinition) -> bool {
        check_network!(network, self);
        self.inner.getAlignCorners()
    }
}

impl<'network> CastLayer<'network> {
    /// See [nvinfer1::ICastLayer::setToType]
    pub fn set_to_type(&mut self, network: &mut NetworkDefinition<'_>, data_type: DataType) {
        check_network!(network, self);
        self.inner.as_mut().setToType(data_type.into())
    }

    /// See [nvinfer1::ICastLayer::getToType]
    pub fn to_type(&self, network: &NetworkDefinition<'_>) -> DataType {
        check_network!(network, self);
        self.inner.getToType().into()
    }
}

#[cfg(test)]
mod test {
    use trtx_sys::LayerType;

    use crate::{Builder, Logger};

    #[test]
    #[cfg(not(feature = "mock"))]
    fn test_get_layer() {
        let logger = Logger::stderr().unwrap();
        let mut builder = Builder::new(&logger).unwrap();
        let mut network = builder.create_network(0).unwrap();
        let input = network
            .add_input("a", trtx_sys::DataType::kFLOAT, &[1])
            .unwrap();
        let a = network
            .add_activation(&input, trtx_sys::ActivationType::kRELU)
            .unwrap()
            .output(&network, 0)
            .unwrap();
        let b = network
            .add_activation(&a, trtx_sys::ActivationType::kRELU)
            .unwrap()
            .output(&network, 0)
            .unwrap();
        let c = network
            .add_activation(&b, trtx_sys::ActivationType::kRELU)
            .unwrap()
            .output(&network, 0)
            .unwrap();
        a.set_name(&mut network, "Fritz").unwrap();
        b.set_name(&mut network, "Adam").unwrap();
        c.set_name(&mut network, "James").unwrap();

        assert_eq!(
            &network
                .layer(0)
                .unwrap()
                .output(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap(),
            "Fritz"
        );
        assert_eq!(
            &network
                .layer(1)
                .unwrap()
                .output(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap(),
            "Adam"
        );
        assert_eq!(
            &network
                .layer(2)
                .unwrap()
                .output(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap(),
            "James"
        );
        assert_eq!(
            network.layer(2).unwrap().layer_type_dynamic(),
            LayerType::kACTIVATION
        );
        network
            .layer(1)
            .unwrap()
            .set_name(&mut network, "Eva")
            .unwrap();
        assert_eq!(
            &network
                .layer(1)
                .unwrap()
                .output(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap(),
            &network
                .layer(2)
                .unwrap()
                .input(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap(),
        );
        assert_eq!(
            "Adam",
            &network
                .layer(2)
                .unwrap()
                .input(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap()
        );
        assert_eq!(&network.layer(1).unwrap().name(&network), "Eva");
    }

    #[test]
    #[cfg(not(feature = "mock"))]
    fn test_inputs_outputs_iter() {
        let logger = Logger::stderr().unwrap();
        let mut builder = Builder::new(&logger).unwrap();
        let mut network = builder.create_network(0).unwrap();
        let a = network
            .add_input("input_a", trtx_sys::DataType::kFLOAT, &[1])
            .unwrap();
        let b = network
            .add_input("input_b", trtx_sys::DataType::kFLOAT, &[1])
            .unwrap();
        let out = network
            .add_elementwise(&a, &b, trtx_sys::ElementWiseOperation::kSUM)
            .unwrap()
            .output(&network, 0)
            .unwrap();
        out.set_name(&mut network, "output_c").unwrap();
        network.mark_output(&out);

        let input_names: Vec<_> = network
            .inputs()
            .map(|t| t.name(&network).unwrap())
            .collect();
        assert_eq!(input_names, ["input_a", "input_b"]);
        assert_eq!(network.inputs().len(), 2);

        let output_names: Vec<_> = network
            .outputs()
            .map(|t| t.name(&network).unwrap())
            .collect();
        assert_eq!(output_names, ["output_c"]);
        assert_eq!(network.outputs().len(), 1);

        // equivalent to the old loop pattern
        let mut old_style = Vec::new();
        for i in 0..network.nb_inputs() {
            old_style.push(network.input(i).unwrap().name(&network).unwrap());
        }
        assert_eq!(input_names, old_style);
    }
}
