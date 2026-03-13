//! Network definition for building TensorRT engines

use crate::interfaces::RecordError;
use cxx::UniquePtr;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::pin::Pin;
use trtx_sys::nvinfer1::{IConcatenationLayer, INetworkDefinition, ITensor};
use trtx_sys::{nvinfer1, LayerType};
use trtx_sys::{ConcreteTrtLayer, TrtLayer};
use trtx_sys::{DataType, MatrixOperation, ScaleMode, TopKOperation};

/// Panics if the layer or tensor was created from a different network.
#[macro_export]
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

use crate::error::{Error, Result};
use crate::interfaces::ErrorRecorder;

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
    pub(crate) inner: *mut nvinfer1::ITensor,
    pub(crate) network: &'network nvinfer1::INetworkDefinition,
}
impl Tensor<'_> {
    pub(crate) unsafe fn new(
        network: *const nvinfer1::INetworkDefinition,
        ptr: *mut nvinfer1::ITensor,
    ) -> Result<Self> {
        unsafe {
            if ptr.is_null() {
                return Err(Error::GetTensorFailed);
            }
            Ok(Self {
                inner: ptr,
                network: network.as_ref().unwrap(),
            })
        }
    }

    #[allow(clippy::mut_from_ref)]
    fn pin_mut(&self) -> Pin<&mut nvinfer1::ITensor> {
        unsafe { Pin::new_unchecked(self.inner.as_mut().unwrap()) }
    }
    fn as_ref(&self) -> &nvinfer1::ITensor {
        unsafe { self.inner.as_ref().unwrap() }
    }
    #[allow(clippy::mut_from_ref)]
    fn as_mut(&self) -> &mut nvinfer1::ITensor {
        unsafe { self.inner.as_mut().unwrap() }
    }
}

pub struct Layer<'network, Inner: TrtLayer> {
    pub(crate) inner: Pin<&'network mut Inner>,
    pub(crate) network: *const nvinfer1::INetworkDefinition,
}

impl<'network, Inner: ConcreteTrtLayer> Layer<'network, Inner> {
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

impl<'network, Inner: TrtLayer> Layer<'network, Inner> {
    /// See [nvinfer1::ILayer::getInput]
    pub fn get_input(
        &self,
        network: &'_ NetworkDefinition,
        index: i32,
    ) -> Result<Tensor<'network>> {
        check_network!(network, self);
        let tensor = self.inner.as_layer().getInput(index);
        unsafe { Tensor::new(self.network, tensor) }
    }

    /// See [nvinfer1::ILayer::getOutput]
    pub fn get_output(
        &self,
        network: &'_ NetworkDefinition,
        index: i32,
    ) -> Result<Tensor<'network>> {
        check_network!(network, self);
        let tensor = self.inner.as_layer().getOutput(index);
        unsafe { Tensor::new(self.network, tensor) }
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
    pub fn set_reshape_dimensions(
        &mut self,
        network: &mut NetworkDefinition,
        dims: &[i64],
    ) -> Result<()> {
        crate::check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(dims);
        self.inner.as_mut().setReshapeDimensions(&dims_obj);
        Ok(())
    }

    pub fn set_first_transpose(
        &mut self,
        network: &mut NetworkDefinition,
        order: &[i32],
    ) -> Result<()> {
        crate::check_network!(network, self);
        let mut order_arr = [0i32; 8];
        let n = order.len().min(8);
        order_arr[..n].copy_from_slice(&order[..n]);
        let perm = trtx_sys::nvinfer1::Permutation { order: order_arr };
        self.inner.as_mut().setFirstTranspose(perm);
        Ok(())
    }
}

impl ResizeLayer<'_> {
    pub fn set_output_dimensions(&mut self, network: &mut NetworkDefinition, dims: &[i64]) {
        crate::check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(dims);
        self.inner.as_mut().setOutputDimensions(&dims_obj);
    }
    pub fn set_resize_mode(&mut self, network: &mut NetworkDefinition, mode: trtx_sys::ResizeMode) {
        crate::check_network!(network, self);
        self.inner.as_mut().setResizeMode(mode.into());
    }
}

impl GatherLayer<'_> {
    pub fn set_gather_mode(&mut self, network: &mut NetworkDefinition, mode: trtx_sys::GatherMode) {
        crate::check_network!(network, self);
        self.inner.as_mut().setMode(mode.into());
    }
}

impl<'network> ScatterLayer<'network> {
    pub fn set_scatter_mode(
        &mut self,
        network: &mut NetworkDefinition,
        mode: trtx_sys::ScatterMode,
    ) {
        crate::check_network!(network, self);
        self.inner.as_mut().setMode(mode.into());
    }
    pub fn set_axis(&mut self, network: &'_ mut NetworkDefinition, axis: i32) {
        crate::check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }
}

impl<'network> ConvolutionLayer<'network> {
    pub fn set_stride(&mut self, network: &mut NetworkDefinition, stride: &[i64; 2]) {
        crate::check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(stride);
        self.inner.as_mut().setStrideNd(&dims_obj);
    }
    pub fn set_padding(&mut self, network: &mut NetworkDefinition, padding: &[i64; 2]) {
        crate::check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(padding);
        self.inner.as_mut().setPaddingNd(&dims_obj);
    }
    pub fn set_dilation(&mut self, network: &mut NetworkDefinition, dilation: &[i64; 2]) {
        crate::check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(dilation);
        self.inner.as_mut().setDilationNd(&dims_obj);
    }
    pub fn set_num_groups(&mut self, network: &mut NetworkDefinition, num_groups: i64) {
        crate::check_network!(network, self);
        self.inner.as_mut().setNbGroups(num_groups);
    }

    /// Set an input tensor by index. Input 0 is the activation; 1 is the kernel tensor; 2 is the bias tensor.
    /// When using input 1 or 2, the layer must have been created with empty weights for that slot.
    pub fn set_input(
        &mut self,
        network: &mut NetworkDefinition,
        index: i32,
        tensor: &'_ Tensor<'network>,
    ) -> Result<()> {
        crate::check_network!(network, self);
        crate::check_network!(network, tensor);
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ILayer>(
                self.inner.as_mut().get_unchecked_mut() as *mut _ as *mut _,
            );
            layer_pin.as_mut().setInput(index, tensor.pin_mut());
        }
        Ok(())
    }
}

impl<'network> DeconvolutionLayer<'network> {
    pub fn set_stride(&mut self, network: &mut NetworkDefinition, stride: &[i64; 2]) -> Result<()> {
        crate::check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(stride);
        self.inner.as_mut().setStrideNd(&dims_obj);
        Ok(())
    }

    /// Set pre-padding (trim this many elements at the start of each spatial dimension of the output).
    /// Pass [pre_h, pre_w] for 2D deconv; TensorRT applies to the spatial dimensions only.
    pub fn set_pre_padding(
        &mut self,
        network: &mut NetworkDefinition,
        padding: &[i64; 2],
    ) -> Result<()> {
        crate::check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(padding);
        self.inner.as_mut().setPrePadding(&dims_obj);
        Ok(())
    }
    /// Set post-padding (trim this many elements at the end of each spatial dimension of the output).
    /// Pass [post_h, post_w] for 2D deconv; TensorRT applies to the spatial dimensions only.
    pub fn set_post_padding(
        &mut self,
        network: &mut NetworkDefinition,
        padding: &[i64; 2],
    ) -> Result<()> {
        crate::check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(padding);
        self.inner.as_mut().setPostPadding(&dims_obj);
        Ok(())
    }
    pub fn set_dilation(
        &mut self,
        network: &mut NetworkDefinition,
        dilation: &[i64; 2],
    ) -> Result<()> {
        crate::check_network!(network, self);
        let dims_obj = trtx_sys::Dims::from_slice(dilation);
        self.inner.as_mut().setDilationNd(&dims_obj);
        Ok(())
    }

    pub fn set_num_groups(
        &mut self,
        network: &mut NetworkDefinition,
        num_groups: i64,
    ) -> Result<()> {
        crate::check_network!(network, self);
        self.inner.as_mut().setNbGroups(num_groups);
        Ok(())
    }
    /// Set an input tensor by index. Input 0 is the activation; 1 is the kernel tensor; 2 is the bias tensor.
    /// When using input 1 or 2, the layer must have been created with empty weights for that slot.
    pub fn set_input(
        &mut self,
        network: &mut NetworkDefinition,
        index: i32,
        tensor: &'_ Tensor<'network>,
    ) -> Result<()> {
        crate::check_network!(network, self);
        crate::check_network!(network, tensor);
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ILayer>(
                self.inner.as_mut().get_unchecked_mut() as *mut _ as *mut _,
            );
            layer_pin.as_mut().setInput(index, tensor.pin_mut());
        }
        Ok(())
    }
}

impl ConcatenationLayer<'_> {
    pub fn set_axis(&mut self, network: &mut NetworkDefinition, axis: i32) {
        crate::check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }
}
impl NormalizationLayer<'_> {
    pub fn set_epsilon(&mut self, network: &mut NetworkDefinition, eps: f32) {
        crate::check_network!(network, self);
        self.inner.as_mut().setEpsilon(eps);
    }
    pub fn get_epsilon(&self, network: &NetworkDefinition) -> f32 {
        crate::check_network!(network, self);
        self.inner.as_ref().getEpsilon()
    }
    pub fn set_axes(&mut self, network: &mut NetworkDefinition, axes: crate::Axes) {
        crate::check_network!(network, self);
        self.inner.as_mut().setAxes(axes.to_bits());
    }
    pub fn get_axes(&self, network: &NetworkDefinition) -> crate::Axes {
        crate::check_network!(network, self);
        crate::Axes::from_bits(self.inner.as_ref().getAxes())
    }
    pub fn set_num_groups(&mut self, network: &mut NetworkDefinition, groups: i64) {
        crate::check_network!(network, self);
        self.inner.as_mut().setNbGroups(groups);
    }
    pub fn get_num_groups(&self, network: &NetworkDefinition) -> i64 {
        crate::check_network!(network, self);
        self.inner.as_ref().getNbGroups()
    }
    pub fn set_compute_precision(&mut self, network: &mut NetworkDefinition, data_type: DataType) {
        crate::check_network!(network, self);
        self.inner.as_mut().setComputePrecision(data_type.into());
    }
    pub fn get_compute_precision(&self, network: &NetworkDefinition) -> DataType {
        crate::check_network!(network, self);
        self.inner.as_ref().getComputePrecision().into()
    }
    pub fn is_v2(&self, network: &NetworkDefinition) -> bool {
        crate::check_network!(network, self);
        self.inner.as_ref().isV2()
    }
}

impl Tensor<'_> {
    pub fn name(&self, network: &NetworkDefinition) -> Result<String> {
        crate::check_network!(network, self);
        let name_ptr = self.as_ref().getName();
        if name_ptr.is_null() {
            return Err(Error::Runtime("Failed to get tensor name".to_string()));
        }
        unsafe { Ok(std::ffi::CStr::from_ptr(name_ptr).to_str()?.to_string()) }
    }

    pub fn set_name(&self, network: &'_ mut NetworkDefinition, name: &str) -> Result<()> {
        crate::check_network!(network, self);
        let name_cstr = std::ffi::CString::new(name)?;
        unsafe {
            self.pin_mut().setName(name_cstr.as_ptr());
        }
        Ok(())
    }

    pub fn dimensions(&self, network: &NetworkDefinition) -> Result<Vec<i64>> {
        crate::check_network!(network, self);
        let result = self.as_ref().getDimensions();
        Ok(result.d[..result.nbDims as usize].to_vec())
    }

    pub fn get_type(&self, network: &NetworkDefinition) -> DataType {
        crate::check_network!(network, self);
        self.as_ref().getType().into()
    }

    /// Set allowed tensor formats (bitmask of TensorFormat). E.g. 1u32 << TensorFormat::kHWC for channels-last.
    /// TensorRT may insert reformat layers when connecting tensors with different formats.
    pub fn set_allowed_formats(
        &mut self,
        network: &mut NetworkDefinition,
        formats: u32,
    ) -> Result<()> {
        crate::check_network!(network, self);
        self.pin_mut().setAllowedFormats(formats);
        Ok(())
    }
}

/// Network definition for building TensorRT engines
pub struct NetworkDefinition<'builder> {
    //pub(crate) inner: Mutex<Pin<&'builder mut INetworkDefinition>>,
    pub(crate) inner: UniquePtr<INetworkDefinition>,
    //error_recorder: Option<Rc<RefCell<ErrorRecorder>>>,
    _builder: PhantomData<&'builder trtx_sys::nvinfer1::IBuilder>,
    small_copied_weights: Vec<Vec<u8>>, // for convenience we hold pointers to scalars here
    error_recorder: Option<Pin<Box<ErrorRecorder>>>,
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
        dims: &[i32],
    ) -> Result<Tensor<'network>> {
        let name_cstr = std::ffi::CString::new(name)?;
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let dims_struct = trtx_sys::Dims::from_slice(&dims_i64);
        let tensor_ptr = unsafe {
            self.inner
                .pin_mut()
                .addInput(name_cstr.as_ptr(), data_type.into(), &dims_struct)
        };
        unsafe { Tensor::new(self.inner.as_ptr(), tensor_ptr) }
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::markOutput`].
    pub fn mark_output(&mut self, tensor: &'_ Tensor) {
        crate::check_network!(self, tensor);
        self.inner.pin_mut().markOutput(tensor.pin_mut());
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::markDebug`].
    /// Mark a tensor for debugging; [IExecutionContext::setDebugListener] will receive it during execution.
    pub fn mark_tensor_debug(&mut self, tensor: &'_ Tensor) -> Result<()> {
        crate::check_network!(self, tensor);
        let success = self.inner.pin_mut().markDebug(tensor.pin_mut());
        if success {
            Ok(())
        } else {
            Err(Error::Runtime("markDebug failed".to_string()))
        }
    }
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::isDebugTensor`].
    /// Mark a tensor for debugging; [nvinfer1::IExecutionContext::setDebugListener] will receive it during execution.
    pub fn is_debug_tensor(&self, tensor: &'_ Tensor) -> bool {
        crate::check_network!(self, tensor);
        self.inner.isDebugTensor(tensor.as_ref())
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getNbInputs`].
    pub fn get_nb_inputs(&self) -> i32 {
        self.inner.getNbInputs()
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getNbOutputs`].
    pub fn get_nb_outputs(&self) -> i32 {
        self.inner.getNbOutputs()
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getInput`].
    pub fn get_input(&self, index: i32) -> Result<Tensor<'network>> {
        let tensor_ptr = self.inner.getInput(index);
        if tensor_ptr.is_null() {
            return Err(Error::Runtime(format!(
                "Failed to get input at index {}",
                index
            )));
        }
        unsafe { Tensor::new(self.inner.as_ptr(), tensor_ptr) }
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getOutput`].
    pub fn get_output(&self, index: i32) -> Result<Tensor<'network>> {
        let tensor_ptr = self.inner.getOutput(index);
        if tensor_ptr.is_null() {
            return Err(Error::Runtime(format!(
                "Failed to get output at index {}",
                index
            )));
        }
        unsafe { Tensor::new(self.inner.as_ptr(), tensor_ptr) }
    }

    /// Number of layers in the network (for introspection/dumping).
    pub fn get_nb_layers(&self) -> i32 {
        self.inner.getNbLayers()
    }

    pub fn get_layer(&self, layer_index: i32) -> Result<DynLayer<'network>> {
        let layer_ptr = self.inner.getLayer(layer_index);
        DynLayer::new_dyn(self.inner.as_ptr(), layer_ptr)
    }

    /// Layer name at index (for introspection/dumping). Returns "(Unnamed)" if null.
    #[deprecated = "use network.get_layer(index)?.name(&network)"]
    pub fn get_layer_name(&self, layer_index: i32) -> Result<String> {
        let layer_ptr = self.inner.getLayer(layer_index);
        unsafe { layer_ptr.as_mut() }
            .ok_or_else(|| Error::Runtime(format!("No layer at index {}", layer_index)))?;
        let name_ptr = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ILayer>(layer_ptr as *mut _)
                .getName()
        };
        Ok(if name_ptr.is_null() {
            "(Unnamed)".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(name_ptr) }
                .to_str()
                .map_err(|e| Error::Runtime(e.to_string()))?
                .to_string()
        })
    }

    /// Layer type enum value at index (for introspection/dumping). See TensorRT LayerType.
    #[deprecated = "use network.get_layer(index)?.layer_type_dyn()"]
    pub fn get_layer_type(&self, layer_index: i32) -> Result<i32> {
        let layer_ptr = self.inner.getLayer(layer_index);
        if layer_ptr.is_null() {
            return Err(Error::Runtime(format!("No layer at index {}", layer_index)));
        }
        let layer_type = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ILayer>(layer_ptr as *mut _)
                .getType()
        };
        Ok(layer_type as i32)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addActivation`].
    pub fn add_activation(
        &mut self,
        input: &'_ Tensor,
        activation_type: trtx_sys::ActivationType,
    ) -> Result<ActivationLayer<'network>> {
        crate::check_network!(self, input);
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
        crate::check_network!(self, input);
        let layer_ptr = self.inner.pin_mut().addUnary(input.pin_mut(), op.into());
        UnaryLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addIdentity`].
    pub fn add_identity(&mut self, input: &'_ Tensor) -> Result<IdentityLayer<'network>> {
        crate::check_network!(self, input);
        let layer_ptr = self.inner.pin_mut().addIdentity(input.pin_mut());

        IdentityLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addCast`].
    pub fn add_cast(
        &mut self,
        input: &'_ Tensor,
        to_type: trtx_sys::DataType,
    ) -> Result<CastLayer<'network>> {
        crate::check_network!(self, input);
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
        crate::check_network!(self, input1);
        crate::check_network!(self, input2);
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
        crate::check_network!(self, input);
        let window_dims = trtx_sys::Dims::new_2d(window_size[0], window_size[1]);
        let layer_ptr =
            self.inner
                .pin_mut()
                .addPoolingNd(input.pin_mut(), pooling_type.into(), &window_dims);
        PoolingLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addShuffle`].
    pub fn add_shuffle(&'_ mut self, input: &'_ Tensor) -> Result<ShuffleLayer<'network>> {
        crate::check_network!(self, input);
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
        crate::check_network!(self, input0);
        crate::check_network!(self, input1);
        let layer_ptr = self.inner.pin_mut().addMatrixMultiply(
            input0.pin_mut(),
            op0.into(),
            input1.pin_mut(),
            op1.into(),
        );
        MatrixMultiplyLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConvolutionNd`].
    pub fn add_convolution(
        &'_ mut self,
        input: &'_ Tensor,
        nb_output_maps: i32,
        kernel_size: &[i32; 2],
        weights: &ConvWeights<'network>,
    ) -> Result<ConvolutionLayer<'network>> {
        crate::check_network!(self, input);
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
        crate::check_network!(self, input);
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

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConcatenation`].
    pub fn add_concatenation(&self, inputs: &[&'_ Tensor]) -> Result<ConcatenationLayer<'network>> {
        for t in inputs.iter() {
            crate::check_network!(self, t);
        }
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
        unsafe { self.add_constant_unsafe(dims, weights, data_type, true) }
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
        unsafe { self.add_constant_unsafe(dims, weights, data_type, false) }
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addSoftMax`].
    pub fn add_softmax(
        &mut self,
        input: &'_ Tensor,
        axes: crate::Axes,
    ) -> Result<SoftMaxLayer<'network>> {
        crate::check_network!(self, input);
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
        crate::check_network!(self, input);
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
        op: trtx_sys::nvinfer1::ReduceOperation,
        axes: crate::Axes,
        keep_dims: bool,
    ) -> Result<ReduceLayer<'network>> {
        crate::check_network!(self, input);
        let axes_bits = axes.to_bits();
        let layer_ptr = self
            .inner
            .pin_mut()
            .addReduce(input.pin_mut(), op, axes_bits, keep_dims);
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
        crate::check_network!(self, input);
        let axis_bytes = axis.to_le_bytes();
        let axis_constant =
            self.add_small_constant_copied(&[], &axis_bytes, trtx_sys::DataType::kINT32)?;
        let axis_tensor = axis_constant.get_output(self, 0)?;
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
        crate::check_network!(self, input);
        crate::check_network!(self, axis_tensor);
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
        crate::check_network!(self, input);
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
        crate::check_network!(self, input);
        let axes_bits = axes.to_bits();
        let layer_ptr = self
            .inner
            .pin_mut()
            .addTopK(input.pin_mut(), op.into(), k, axes_bits);
        TopKLayer::new(self.inner.as_ptr(), layer_ptr)
    }
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addResize`].
    pub fn add_resize(&mut self, input: &'_ Tensor) -> Result<ResizeLayer<'network>> {
        crate::check_network!(self, input);
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
        crate::check_network!(self, data);
        crate::check_network!(self, indices);
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
        crate::check_network!(self, data);
        crate::check_network!(self, indices);
        crate::check_network!(self, updates);
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
        crate::check_network!(self, input);
        crate::check_network!(self, scale);
        let layer_ptr =
            self.inner
                .pin_mut()
                .addQuantize(input.pin_mut(), scale.pin_mut(), output_type.into());
        QuantizeLayer::new(self.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addDequantize`].
    pub fn add_dequantize(
        &mut self,
        input: &'_ Tensor,
        scale: &'_ Tensor,
        output_type: trtx_sys::DataType,
    ) -> Result<DequantizeLayer<'network>> {
        crate::check_network!(self, input);
        crate::check_network!(self, scale);
        let layer_ptr = self.inner.pin_mut().addDequantize(
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
        crate::check_network!(self, condition);
        crate::check_network!(self, then_input);
        crate::check_network!(self, else_input);
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
        crate::check_network!(self, input);
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
        crate::check_network!(self, condition);
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
        crate::check_network!(self, query);
        crate::check_network!(self, key);
        crate::check_network!(self, value);
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
}

// --- Attention: get_output ---

impl<'network> Attention<'network> {
    /// See [`trtx_sys::nvinfer1::IAttention::getOutput`]. IAttention has one output (index 0).
    pub fn get_output(&self, network: &NetworkDefinition, index: i32) -> Result<Tensor<'network>> {
        crate::check_network!(network, self);
        let tensor_ptr = self.inner.getOutput(index);
        unsafe { Tensor::new(self.network, tensor_ptr) }
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
        crate::check_network!(network, self);
        crate::check_network!(network, initial_value);
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
        crate::check_network!(network, self);
        crate::check_network!(network, tensor);
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
        crate::check_network!(network, self);
        crate::check_network!(network, tensor);
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
        output_kind: trtx_sys::nvinfer1::LoopOutput,
        axis: i32,
    ) -> Result<LoopOutputLayer<'network>> {
        crate::check_network!(network, self);
        crate::check_network!(network, tensor);
        let layer_ptr = self
            .inner
            .as_mut()
            .addLoopOutput(tensor.pin_mut(), output_kind, axis);
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
        crate::check_network!(network, self);
        crate::check_network!(network, condition);
        let layer_ptr = self.inner.as_mut().setCondition(condition.pin_mut());
        ConditionLayer::new(network.inner.as_ptr(), layer_ptr)
    }

    /// See [`trtx_sys::nvinfer1::IIfConditional::addInput`].
    pub fn add_input(
        &mut self,
        network: &mut NetworkDefinition,
        input: &'_ Tensor,
    ) -> Result<IfConditionalInputLayer<'network>> {
        crate::check_network!(network, self);
        crate::check_network!(network, input);
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
        crate::check_network!(network, self);
        crate::check_network!(network, true_output);
        crate::check_network!(network, false_output);
        let layer_ptr = self
            .inner
            .as_mut()
            .addOutput(true_output.pin_mut(), false_output.pin_mut());
        IfConditionalOutputLayer::new(network.inner.as_ptr(), layer_ptr)
    }
}

// --- RecurrenceLayer: set_input(1, tensor) for value from inside loop ---

impl<'network> RecurrenceLayer<'network> {
    /// See [`trtx_sys::nvinfer1::IRecurrenceLayer`]. Input 0 = initial value (set at creation); input 1 = value from previous iteration (from inside loop).
    pub fn set_input(
        &mut self,
        network: &mut NetworkDefinition,
        index: i32,
        tensor: &'_ Tensor<'network>,
    ) -> Result<()> {
        crate::check_network!(network, self);
        crate::check_network!(network, tensor);
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ILayer>(
                self.inner.as_mut().get_unchecked_mut() as *mut _ as *mut _,
            );
            layer_pin.as_mut().setInput(index, tensor.pin_mut());
        }
        Ok(())
    }
}

// --- IteratorLayer: set_axis, set_reverse ---

impl IteratorLayer<'_> {
    /// See [`trtx_sys::nvinfer1::IIteratorLayer::setAxis`].
    pub fn set_axis(&mut self, network: &mut NetworkDefinition, axis: i32) {
        crate::check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }
    /// See [`trtx_sys::nvinfer1::IIteratorLayer::setReverse`].
    pub fn set_reverse(&mut self, network: &mut NetworkDefinition, reverse: bool) {
        crate::check_network!(network, self);
        self.inner.as_mut().setReverse(reverse);
    }
}

// --- LoopOutputLayer: get_loop_output, set_axis (for concatenation), set_input for index 1 ---

impl LoopOutputLayer<'_> {
    /// See [`trtx_sys::nvinfer1::ILoopOutputLayer::getLoopOutput`].
    pub fn get_loop_output(&self, network: &NetworkDefinition) -> trtx_sys::nvinfer1::LoopOutput {
        crate::check_network!(network, self);
        self.inner.as_ref().getLoopOutput()
    }
    /// See [`trtx_sys::nvinfer1::ILoopOutputLayer::setAxis`]. Ignored if output kind is kLAST_VALUE.
    pub fn set_axis(&mut self, network: &mut NetworkDefinition, axis: i32) {
        crate::check_network!(network, self);
        self.inner.as_mut().setAxis(axis);
    }
}

// --- TripLimitLayer: get_trip_limit (getter only) ---

impl TripLimitLayer<'_> {
    /// See [`trtx_sys::nvinfer1::ITripLimitLayer::getTripLimit`].
    pub fn get_trip_limit(&self, network: &NetworkDefinition) -> trtx_sys::nvinfer1::TripLimit {
        crate::check_network!(network, self);
        self.inner.as_ref().getTripLimit()
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
        crate::check_network!(self, input);
        crate::check_network!(self, scale);
        crate::check_network!(self, bias);
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
        crate::check_network!(self, input);
        crate::check_network!(self, scale);
        crate::check_network!(self, bias);
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
            .get_output(&network, 0)
            .unwrap();
        let b = network
            .add_activation(&a, trtx_sys::ActivationType::kRELU)
            .unwrap()
            .get_output(&network, 0)
            .unwrap();
        let c = network
            .add_activation(&b, trtx_sys::ActivationType::kRELU)
            .unwrap()
            .get_output(&network, 0)
            .unwrap();
        a.set_name(&mut network, "Fritz").unwrap();
        b.set_name(&mut network, "Adam").unwrap();
        c.set_name(&mut network, "James").unwrap();

        assert_eq!(
            &network
                .get_layer(0)
                .unwrap()
                .get_output(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap(),
            "Fritz"
        );
        assert_eq!(
            &network
                .get_layer(1)
                .unwrap()
                .get_output(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap(),
            "Adam"
        );
        assert_eq!(
            &network
                .get_layer(2)
                .unwrap()
                .get_output(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap(),
            "James"
        );
        assert_eq!(
            network.get_layer(2).unwrap().layer_type_dynamic(),
            LayerType::kACTIVATION
        );
        network
            .get_layer(1)
            .unwrap()
            .set_name(&mut network, "Eva")
            .unwrap();
        assert_eq!(
            &network
                .get_layer(1)
                .unwrap()
                .get_output(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap(),
            &network
                .get_layer(2)
                .unwrap()
                .get_input(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap(),
        );
        assert_eq!(
            "Adam",
            &network
                .get_layer(2)
                .unwrap()
                .get_input(&network, 0)
                .unwrap()
                .name(&network)
                .unwrap()
        );
        assert_eq!(&network.get_layer(1).unwrap().name(&network), "Eva");
    }
}
