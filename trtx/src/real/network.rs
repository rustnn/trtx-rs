//! Real TensorRT network implementation
//! No #[cfg] - this module is only compiled when mock feature is disabled

use std::mem::transmute;
use std::pin::Pin;
use std::ptr;
use std::sync::Mutex;
use trtx_sys::nvinfer1::{IConcatenationLayer, INetworkDefinition, ITensor};
use trtx_sys::{DataType, MatrixOperation, ScaleMode, TopKOperation};

use crate::error::{Error, LayerTypeKind, Result};
use crate::network::*;
use trtx_sys::nvinfer1::ILayer;

/// Macro to implement Layer trait for real TensorRT types
macro_rules! impl_layer_real {
    ($name:ident, $trt_type:path) => {
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
                    transmute::<&mut Pin<&mut $trt_type>, &mut Pin<&mut ILayer>>(lock)
                        .as_mut()
                        .setName(name_cstr.as_ptr())
                };
                Ok(())
            }
        }
    };
}

impl_layer_real!(ShuffleLayer, trtx_sys::nvinfer1::IShuffleLayer);
impl_layer_real!(ActivationLayer, trtx_sys::nvinfer1::IActivationLayer);
impl_layer_real!(ElementWiseLayer, trtx_sys::nvinfer1::IElementWiseLayer);
impl_layer_real!(ResizeLayer, trtx_sys::nvinfer1::IResizeLayer);
impl_layer_real!(TopKLayer, trtx_sys::nvinfer1::ITopKLayer);
impl_layer_real!(GatherLayer, trtx_sys::nvinfer1::IGatherLayer);
impl_layer_real!(ScatterLayer, trtx_sys::nvinfer1::IScatterLayer);
impl_layer_real!(SelectLayer, trtx_sys::nvinfer1::ISelectLayer);
impl_layer_real!(
    MatrixMultiplyLayer,
    trtx_sys::nvinfer1::IMatrixMultiplyLayer
);
impl_layer_real!(SoftMaxLayer, trtx_sys::nvinfer1::ISoftMaxLayer);
impl_layer_real!(ReduceLayer, trtx_sys::nvinfer1::IReduceLayer);
impl_layer_real!(CumulativeLayer, trtx_sys::nvinfer1::ICumulativeLayer);
impl_layer_real!(PoolingLayer, trtx_sys::nvinfer1::IPoolingLayer);
impl_layer_real!(ConvolutionLayer, trtx_sys::nvinfer1::IConvolutionLayer);
impl_layer_real!(DeconvolutionLayer, trtx_sys::nvinfer1::IDeconvolutionLayer);
impl_layer_real!(QuantizeLayer, trtx_sys::nvinfer1::IQuantizeLayer);
impl_layer_real!(DequantizeLayer, trtx_sys::nvinfer1::IDequantizeLayer);
impl_layer_real!(ConstantLayer, trtx_sys::nvinfer1::IConstantLayer);
impl_layer_real!(ConcatenationLayer, trtx_sys::nvinfer1::IConcatenationLayer);
impl_layer_real!(ScaleLayer, trtx_sys::nvinfer1::IScaleLayer);
impl_layer_real!(SliceLayer, trtx_sys::nvinfer1::ISliceLayer);
impl_layer_real!(UnaryLayer, trtx_sys::nvinfer1::IUnaryLayer);
impl_layer_real!(IdentityLayer, trtx_sys::nvinfer1::IIdentityLayer);
impl_layer_real!(PaddingLayer, trtx_sys::nvinfer1::IPaddingLayer);
impl_layer_real!(CastLayer, trtx_sys::nvinfer1::ICastLayer);

// Those are actually not ILayers in C++ but just ICopy

impl ShuffleLayer<'_> {
    pub fn set_reshape_dimensions(&mut self, dims: &[i32]) -> Result<()> {
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
        self.inner
            .lock()
            .unwrap()
            .as_mut()
            .setReshapeDimensions(&dims_obj);
        Ok(())
    }

    pub fn set_first_transpose(&mut self, order: &[i32]) -> Result<()> {
        let mut order_arr = [0i32; 8];
        let n = order.len().min(8);
        order_arr[..n].copy_from_slice(&order[..n]);
        let perm = trtx_sys::nvinfer1::Permutation { order: order_arr };
        self.inner.lock().unwrap().as_mut().setFirstTranspose(perm);
        Ok(())
    }
}

impl ResizeLayer<'_> {
    pub fn set_output_dimensions(&mut self, dims: &[i32]) {
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
        self.inner
            .lock()
            .unwrap()
            .as_mut()
            .setOutputDimensions(&dims_obj);
    }
    pub fn set_resize_mode(&mut self, mode: trtx_sys::ResizeMode) {
        self.inner.lock().unwrap().as_mut().setResizeMode(mode);
    }
}

impl GatherLayer<'_> {
    pub fn set_gather_mode(&mut self, mode: trtx_sys::nvinfer1::GatherMode) {
        self.inner.lock().unwrap().as_mut().setMode(mode);
    }
}

impl ScatterLayer<'_> {
    pub fn set_scatter_mode(&mut self, mode: trtx_sys::nvinfer1::ScatterMode) {
        self.inner.lock().unwrap().as_mut().setMode(mode);
    }
    pub fn set_axis(&mut self, axis: i32) {
        self.inner.lock().unwrap().as_mut().setAxis(axis);
    }
}

impl ConvolutionLayer<'_> {
    pub fn set_stride(&self, stride: &[i32; 2]) {
        let dims_i64: Vec<i64> = stride.iter().map(|&s| s as i64).collect();
        let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
        self.inner.lock().unwrap().as_mut().setStrideNd(&dims_obj);
    }
    pub fn set_padding(&self, padding: &[i32; 2]) {
        let dims_i64: Vec<i64> = padding.iter().map(|&p| p as i64).collect();
        let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
        self.inner.lock().unwrap().as_mut().setPaddingNd(&dims_obj);
    }
    pub fn set_dilation(&self, dilation: &[i32; 2]) {
        let dims_i64: Vec<i64> = dilation.iter().map(|&d| d as i64).collect();
        let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
        self.inner.lock().unwrap().as_mut().setDilationNd(&dims_obj);
    }
    pub fn set_num_groups(&self, num_groups: i32) {
        self.inner
            .lock()
            .unwrap()
            .as_mut()
            .setNbGroups(num_groups as i64);
    }

    /// Set an input tensor by index. Input 0 is the activation; 1 is the kernel tensor; 2 is the bias tensor.
    /// When using input 1 or 2, the layer must have been created with empty weights for that slot.
    pub fn set_input(&self, index: i32, tensor: &Tensor) -> Result<()> {
        let mut lock = self.inner.lock()?;
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ILayer>(
                lock.as_mut().get_unchecked_mut() as *mut _ as *mut _,
            );
            layer_pin
                .as_mut()
                .setInput(index, tensor.inner.lock()?.as_mut());
        }
        Ok(())
        //(self.inner.get_mut()?.as_mut().get_unchecked_mut()  as *mut ILayer)
        //.setInput(index, tensor.inner.get_mut());
        //Ok(())
    }
}

impl DeconvolutionLayer<'_> {
    pub fn set_stride(&mut self, stride: &[i32; 2]) -> Result<()> {
        let dims_i64: Vec<i64> = stride.iter().map(|&p| p as i64).collect();
        let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
        self.inner.get_mut()?.as_mut().setStrideNd(&dims_obj);
        Ok(())
    }

    /// Set pre-padding (trim this many elements at the start of each spatial dimension of the output).
    /// Pass [pre_h, pre_w] for 2D deconv; TensorRT applies to the spatial dimensions only.
    pub fn set_pre_padding(&mut self, padding: &[i32; 2]) -> Result<()> {
        let dims_i64: Vec<i64> = padding.iter().map(|&p| p as i64).collect();
        let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
        self.inner.get_mut()?.as_mut().setPrePadding(&dims_obj);
        Ok(())
    }
    /// Set post-padding (trim this many elements at the end of each spatial dimension of the output).
    /// Pass [post_h, post_w] for 2D deconv; TensorRT applies to the spatial dimensions only.
    pub fn set_post_padding(&mut self, padding: &[i32; 2]) -> Result<()> {
        let dims_i64: Vec<i64> = padding.iter().map(|&p| p as i64).collect();
        let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
        self.inner.get_mut()?.as_mut().setPostPadding(&dims_obj);
        Ok(())
    }
    pub fn set_dilation(&mut self, dilation: &[i32; 2]) -> Result<()> {
        let dims_i64: Vec<i64> = dilation.iter().map(|&p| p as i64).collect();
        let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
        self.inner.get_mut()?.as_mut().setDilationNd(&dims_obj);
        Ok(())
    }

    pub fn set_num_groups(&mut self, num_groups: i32) -> Result<()> {
        self.inner
            .get_mut()?
            .as_mut()
            .setNbGroups(num_groups as i64);
        Ok(())
    }
    /// Set an input tensor by index. Input 0 is the activation; 1 is the kernel tensor; 2 is the bias tensor.
    /// When using input 1 or 2, the layer must have been created with empty weights for that slot.
    pub fn set_input(&mut self, index: i32, tensor: &Tensor) -> Result<()> {
        let mut lock = self.inner.lock()?;
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ILayer>(
                lock.as_mut().get_unchecked_mut() as *mut _ as *mut _,
            );
            layer_pin
                .as_mut()
                .setInput(index, tensor.inner.lock()?.as_mut());
        }
        Ok(())
    }
}

impl ConcatenationLayer<'_> {
    pub fn set_axis(&self, axis: i32) {
        self.inner.lock().unwrap().as_mut().setAxis(axis);
    }
}

impl Tensor<'_> {
    pub fn name(&self) -> Result<String> {
        let name_ptr = self.inner.lock().unwrap().as_ref().getName();
        if name_ptr.is_null() {
            return Err(Error::Runtime("Failed to get tensor name".to_string()));
        }
        unsafe { Ok(std::ffi::CStr::from_ptr(name_ptr).to_str()?.to_string()) }
    }

    pub fn set_name(&mut self, name: &str) -> Result<()> {
        let name_cstr = std::ffi::CString::new(name)?;
        unsafe {
            self.inner.lock()?.as_mut().setName(name_cstr.as_ptr());
        }
        Ok(())
    }

    pub fn dimensions(&self) -> Result<Vec<i32>> {
        let result = self.inner.lock().unwrap().as_ref().getDimensions();
        Ok(result
            .d
            .iter()
            .take(result.nbDims as usize)
            .map(|&i| i as i32)
            .collect())
    }

    pub fn get_type(&self) -> DataType {
        self.inner.lock().unwrap().as_ref().getType().into()
    }

    /// Set allowed tensor formats (bitmask of TensorFormat). E.g. 1u32 << TensorFormat::kHWC for channels-last.
    /// TensorRT may insert reformat layers when connecting tensors with different formats.
    pub fn set_allowed_formats(&mut self, formats: u32) -> Result<()> {
        self.inner.get_mut()?.as_mut().setAllowedFormats(formats);
        Ok(())
    }
}

/// Network definition for building TensorRT engines
pub struct NetworkDefinition<'builder> {
    pub(crate) inner: Mutex<Pin<&'builder mut INetworkDefinition>>,
}

impl<'builder> NetworkDefinition<'builder> {
    pub(crate) fn from_ptr(ptr: &'builder mut INetworkDefinition) -> Self {
        Self {
            inner: unsafe { Mutex::new(Pin::new_unchecked(ptr)) },
        }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut INetworkDefinition {
        unsafe { self.inner.lock().unwrap().as_mut().get_unchecked_mut() }
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addInput`].
    pub fn add_input(
        &self,
        name: &str,
        data_type: trtx_sys::DataType,
        dims: &[i32],
    ) -> Result<Tensor<'_>> {
        let name_cstr = std::ffi::CString::new(name)?;
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let dims_struct = trtx_sys::Dims::from_slice(&dims_i64);
        let tensor_ptr = unsafe {
            self.inner
                .lock()?
                .as_mut()
                .addInput(name_cstr.as_ptr(), data_type.into(), &dims_struct)
        };
        if tensor_ptr.is_null() {
            return Err(Error::Runtime(format!("Failed to add input: {}", name)));
        }
        Ok(Tensor {
            inner: unsafe { Mutex::new(Pin::new_unchecked(tensor_ptr.as_mut().unwrap())) },
        })
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::markOutput`].
    pub fn mark_output(&self, tensor: &Tensor) {
        self.inner
            .lock()
            .unwrap()
            .as_mut()
            .markOutput(tensor.inner.lock().unwrap().as_mut());
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getNbInputs`].
    pub fn get_nb_inputs(&self) -> i32 {
        self.inner.lock().unwrap().as_ref().getNbInputs()
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getNbOutputs`].
    pub fn get_nb_outputs(&self) -> i32 {
        self.inner.lock().unwrap().as_ref().getNbOutputs()
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getInput`].
    pub fn get_input(&self, index: i32) -> Result<Tensor<'_>> {
        let tensor_ptr = self.inner.lock().unwrap().as_ref().getInput(index);
        if tensor_ptr.is_null() {
            return Err(Error::Runtime(format!(
                "Failed to get input at index {}",
                index
            )));
        }
        Ok(Tensor {
            inner: unsafe { Mutex::new(Pin::new_unchecked(tensor_ptr.as_mut().unwrap())) },
        })
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::getOutput`].
    pub fn get_output(&self, index: i32) -> Result<Tensor<'_>> {
        let tensor_ptr = self.inner.lock().unwrap().as_ref().getOutput(index);
        if tensor_ptr.is_null() {
            return Err(Error::Runtime(format!(
                "Failed to get output at index {}",
                index
            )));
        }
        Ok(Tensor {
            inner: unsafe { Mutex::new(Pin::new_unchecked(tensor_ptr.as_mut().unwrap())) },
        })
    }

    /// Number of layers in the network (for introspection/dumping).
    pub fn get_nb_layers(&self) -> i32 {
        self.inner.lock().unwrap().as_ref().getNbLayers()
    }

    /// Layer name at index (for introspection/dumping). Returns "(Unnamed)" if null.
    pub fn get_layer_name(&self, layer_index: i32) -> Result<String> {
        let layer_ptr = self.inner.lock().unwrap().as_ref().getLayer(layer_index);
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
    pub fn get_layer_type(&self, layer_index: i32) -> Result<i32> {
        let layer_ptr = self.inner.lock()?.getLayer(layer_index);
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
        &self,
        input: &Tensor,
        activation_type: trtx_sys::ActivationType,
    ) -> Result<ActivationLayer<'_>> {
        let layer_ptr = self
            .inner
            .lock()?
            .as_mut()
            .addActivation(input.inner.lock()?.as_mut(), activation_type.into());
        let layer = unsafe { layer_ptr.as_mut() }
            .ok_or(Error::LayerCreationFailed(LayerTypeKind::Activation))?;
        Ok(ActivationLayer::from_ptr(layer))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addUnary`].
    pub fn add_unary(
        &mut self,
        input: &mut Tensor,
        op: trtx_sys::nvinfer1::UnaryOperation,
    ) -> Result<UnaryLayer<'_>> {
        let layer_ptr = self
            .inner
            .lock()
            .unwrap()
            .as_mut()
            .addUnary(input.inner.lock().unwrap().as_mut(), op);
        Ok(UnaryLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Unary))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addIdentity`].
    pub fn add_identity(&mut self, input: &mut Tensor) -> Result<IdentityLayer<'_>> {
        let layer_ptr = self
            .inner
            .lock()
            .unwrap()
            .as_mut()
            .addIdentity(input.inner.lock().unwrap().as_mut());
        Ok(IdentityLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Indentity))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addCast`].
    pub fn add_cast(
        &self,
        input: &mut Tensor,
        to_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<CastLayer<'_>> {
        let layer_ptr = self
            .inner
            .lock()
            .unwrap()
            .as_mut()
            .addCast(input.inner.lock().unwrap().as_mut(), to_type);
        Ok(CastLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Cast))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addElementWise`].
    pub fn add_elementwise(
        &mut self,
        input1: &mut Tensor,
        input2: &mut Tensor,
        op: trtx_sys::nvinfer1::ElementWiseOperation,
    ) -> Result<ElementWiseLayer<'_>> {
        let layer_ptr = self.inner.lock().unwrap().as_mut().addElementWise(
            input1.inner.lock().unwrap().as_mut(),
            input2.inner.lock().unwrap().as_mut(),
            op,
        );
        Ok(ElementWiseLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Elementwise))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addPoolingNd`].
    pub fn add_pooling(
        &mut self,
        input: &Tensor,
        pooling_type: trtx_sys::PoolingType,
        window_size: &[i32; 2],
    ) -> Result<PoolingLayer<'_>> {
        let window_dims = trtx_sys::Dims::new_2d(window_size[0] as i64, window_size[1] as i64);
        let layer_ptr = self.inner.lock().unwrap().as_mut().addPoolingNd(
            input.inner.lock().unwrap().as_mut(),
            pooling_type.into(),
            &window_dims,
        );
        Ok(PoolingLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Pooling))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addShuffle`].
    pub fn add_shuffle(&mut self, input: &mut Tensor) -> Result<ShuffleLayer<'_>> {
        let layer_ptr = self
            .inner
            .lock()
            .unwrap()
            .as_mut()
            .addShuffle(input.inner.lock().unwrap().as_mut());
        Ok(ShuffleLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Shuffle))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addMatrixMultiply`].
    pub fn add_matrix_multiply(
        &mut self,
        input0: &mut Tensor,
        op0: MatrixOperation,
        input1: &mut Tensor,
        op1: MatrixOperation,
    ) -> Result<MatrixMultiplyLayer<'_>> {
        let layer_ptr = self.inner.lock().unwrap().as_mut().addMatrixMultiply(
            input0.inner.lock().unwrap().as_mut(),
            op0.into(),
            input1.inner.lock().unwrap().as_mut(),
            op1.into(),
        );
        Ok(MatrixMultiplyLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::MatrixMultiply))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConvolutionNd`].
    pub fn add_convolution(
        &mut self,
        input: &Tensor,
        nb_output_maps: i32,
        kernel_size: &[i32; 2],
        weights: &ConvWeights<'_>,
    ) -> Result<ConvolutionLayer<'_>> {
        let kernel_dtype = weights.kernel_dtype;
        let kernel_weights = weights.kernel_weights;
        let bias_weights = weights.bias_weights;
        let bias_dtype = weights.bias_dtype;
        let kernel_bpe = match kernel_dtype {
            DataType::kFLOAT => 4,
            DataType::kHALF => 2,
            DataType::kINT8 => 1,
            DataType::kINT32 => 4,
            _ => {
                return Err(Error::Runtime(format!(
                    "Unsupported kernel weight type for convolution: {kernel_dtype:?}",
                )))
            }
        };
        let weight_count = (kernel_weights.len() / kernel_bpe) as i64;
        let bias_dtype_val = bias_dtype.unwrap_or(kernel_dtype);
        let bias_bpe = match bias_dtype_val {
            DataType::kFLOAT => 4,
            DataType::kHALF => 2,
            DataType::kINT8 => 1,
            DataType::kINT32 => 4,
            _ => {
                return Err(Error::Runtime(format!(
                    "Unsupported bias weight type for convolution: {bias_dtype_val:?}",
                )))
            }
        };
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
        let layer_ptr = self.inner.get_mut()?.as_mut().addConvolutionNd(
            input.inner.lock()?.as_mut(),
            nb_output_maps as i64,
            &kernel_dims,
            kernel_w,
            bias_w,
        );
        let layer_ptr = unsafe {
            layer_ptr
                .as_mut()
                .ok_or_else(|| Error::LayerCreationFailed(LayerTypeKind::Convolution))?
        };
        Ok(ConvolutionLayer::from_ptr(layer_ptr))
    }

    /// Add a 2D deconvolution layer. Same input semantics as convolution: input 0 = activation,
    /// input 1 = kernel tensor (use set_input(1, tensor) when kernel_weights is empty),
    /// input 2 = bias tensor (use set_input(2, tensor) when bias_weights is None/empty).
    ///
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addDeconvolutionNd`].
    pub fn add_deconvolution(
        &mut self,
        input: &Tensor,
        nb_output_maps: i32,
        kernel_size: &[i32; 2],
        weights: &ConvWeights<'_>,
    ) -> Result<DeconvolutionLayer<'_>> {
        let kernel_dtype = weights.kernel_dtype;
        let kernel_weights = weights.kernel_weights;
        let bias_weights = weights.bias_weights;
        let bias_dtype = weights.bias_dtype;
        let kernel_bpe = match kernel_dtype {
            DataType::kFLOAT => 4,
            DataType::kHALF => 2,
            DataType::kINT8 => 1,
            DataType::kINT32 => 4,
            _ => {
                return Err(Error::Runtime(format!(
                    "Unsupported kernel weight type for deconvolution: {kernel_dtype:?}",
                )))
            }
        };
        let weight_count = (kernel_weights.len() / kernel_bpe) as i64;
        let bias_dtype_val = bias_dtype.unwrap_or(kernel_dtype);
        let bias_bpe = match bias_dtype_val {
            DataType::kFLOAT => 4,
            DataType::kHALF => 2,
            DataType::kINT8 => 1,
            DataType::kINT32 => 4,
            _ => {
                return Err(Error::Runtime(format!(
                    "Unsupported bias weight type for deconvolution: {bias_dtype:?}",
                )))
            }
        };
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
        let layer_ptr = self.inner.get_mut()?.as_mut().addDeconvolutionNd(
            input.inner.lock()?.as_mut(),
            nb_output_maps as i64,
            kernel_dims,
            kernel_w,
            bias_w,
        );
        Ok(DeconvolutionLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Deconvolution))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConcatenation`].
    pub fn add_concatenation(&self, inputs: &[&Tensor]) -> Result<ConcatenationLayer<'_>> {
        let mut input_ptrs: Vec<*mut std::ffi::c_void> = inputs
            .iter()
            .map(|t| unsafe {
                t.inner.lock().unwrap().as_mut().get_unchecked_mut() as &mut ITensor as *mut ITensor
                    as *mut _
            })
            .collect();
        let layer_ptr = unsafe {
            trtx_sys::network_add_concatenation(
                self.inner.lock()?.as_mut().get_unchecked_mut() as *mut INetworkDefinition
                    as *mut std::ffi::c_void,
                input_ptrs.as_mut_ptr(),
                inputs.len() as i32,
            )
        } as *mut IConcatenationLayer;
        let layer = unsafe { layer_ptr.as_mut() }
            .ok_or(Error::LayerCreationFailed(LayerTypeKind::Concatenation))?;
        Ok(ConcatenationLayer::from_ptr(layer))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addConstant`].
    pub fn add_constant(
        &self,
        dims: &[i32],
        weights: &[u8],
        data_type: trtx_sys::DataType,
    ) -> Result<ConstantLayer<'_>> {
        let element_count: i64 = dims.iter().map(|&d| d as i64).product();
        let bytes_per_element = match data_type {
            DataType::kFLOAT => 4,
            DataType::kHALF => 2,
            DataType::kINT8 => 1,
            DataType::kINT32 => 4,
            DataType::kUINT8 => 1,
            DataType::kBOOL => 1,
            _ => {
                return Err(Error::Runtime(format!(
                    "Unsupported data type: {data_type:?}",
                )))
            }
        };
        let expected_bytes = element_count * bytes_per_element;
        if weights.len() as i64 != expected_bytes {
            return Err(Error::Runtime(format!(
                "Weight size mismatch: expected {} bytes, got {} bytes",
                expected_bytes,
                weights.len()
            )));
        }
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let dims_struct = trtx_sys::Dims::from_slice(&dims_i64);
        let weights_struct = trtx_sys::nvinfer1::Weights::new_with_type(
            data_type.into(),
            weights.as_ptr() as *const std::ffi::c_void,
            element_count,
        );
        let layer_ptr = self
            .inner
            .lock()
            .unwrap()
            .as_mut()
            .addConstant(&dims_struct, weights_struct);
        Ok(ConstantLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Constant))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addSoftMax`].
    pub fn add_softmax(&mut self, input: &Tensor, axes: u32) -> Result<SoftMaxLayer<'_>> {
        let layer_ptr = unsafe {
            self.inner
                .lock()
                .unwrap()
                .as_mut()
                .addSoftMax(input.inner.lock().unwrap().as_mut())
                .as_mut()
        }
        .ok_or(Error::LayerCreationFailed(LayerTypeKind::Softmax))?;

        let rtn = SoftMaxLayer::from_ptr(layer_ptr);
        rtn.inner.lock().unwrap().as_mut().setAxes(axes);
        Ok(rtn)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addScale`].
    pub fn add_scale(
        &mut self,
        input: &mut Tensor,
        mode: ScaleMode,
        shift: &[u8],
        scale: &[u8],
        power: &[u8],
    ) -> Result<ScaleLayer<'_>> {
        let weight_count = match mode {
            ScaleMode::kUNIFORM => 1i64,
            ScaleMode::kCHANNEL => {
                let input_dims = input.dimensions()?;
                if input_dims.len() >= 4 {
                    input_dims[1] as i64
                } else if !input_dims.is_empty() {
                    input_dims[0] as i64
                } else {
                    1i64
                }
            }
            ScaleMode::kELEMENTWISE => {
                let input_dims = input.dimensions()?;
                input_dims.iter().map(|&d| d as i64).product::<i64>()
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
        let layer_ptr = self.inner.lock().unwrap().as_mut().addScale(
            input.inner.lock().unwrap().as_mut(),
            mode.into(),
            shift_w,
            scale_w,
            power_w,
        );
        Ok(ScaleLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Scale))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addReduce`].
    pub fn add_reduce(
        &mut self,
        input: &mut Tensor,
        op: trtx_sys::nvinfer1::ReduceOperation,
        axes: u32,
        keep_dims: bool,
    ) -> Result<ReduceLayer<'_>> {
        let layer_ptr = self.inner.lock().unwrap().as_mut().addReduce(
            input.inner.lock().unwrap().as_mut(),
            op,
            axes,
            keep_dims,
        );
        Ok(ReduceLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Reduce))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addCumulative`].
    pub fn add_cumulative(
        &self,
        input: &mut Tensor,
        axis: i32,
        op: trtx_sys::CumulativeOperation,
        exclusive: bool,
        reverse: bool,
    ) -> Result<CumulativeLayer<'_>> {
        let axis_bytes = axis.to_le_bytes();
        let axis_constant = self.add_constant(&[], &axis_bytes, trtx_sys::DataType::kINT32)?;
        let axis_tensor = axis_constant.get_output(0)?;
        self.add_cumulative_with_axis_tensor(input, &axis_tensor, op, exclusive, reverse)
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addCumulative`].
    pub fn add_cumulative_with_axis_tensor(
        &self,
        input: &Tensor,
        axis_tensor: &Tensor,
        op: trtx_sys::CumulativeOperation,
        exclusive: bool,
        reverse: bool,
    ) -> Result<CumulativeLayer<'_>> {
        let layer_ptr = self.inner.lock().unwrap().as_mut().addCumulative(
            input.inner.lock().unwrap().as_mut(),
            axis_tensor.inner.lock().unwrap().as_mut(),
            op.into(),
            exclusive,
            reverse,
        );
        Ok(CumulativeLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Cumulative))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addSlice`].
    pub fn add_slice(
        &mut self,
        input: &mut Tensor,
        start: &[i32],
        size: &[i32],
        stride: &[i32],
    ) -> Result<SliceLayer<'_>> {
        if start.len() != size.len() || start.len() != stride.len() {
            return Err(Error::Runtime(
                "start, size, and stride must have the same length".to_string(),
            ));
        }
        let start_i64: Vec<i64> = start.iter().map(|&d| d as i64).collect();
        let size_i64: Vec<i64> = size.iter().map(|&d| d as i64).collect();
        let stride_i64: Vec<i64> = stride.iter().map(|&d| d as i64).collect();
        let start_dims = trtx_sys::Dims::from_slice(&start_i64);
        let size_dims = trtx_sys::Dims::from_slice(&size_i64);
        let stride_dims = trtx_sys::Dims::from_slice(&stride_i64);
        let layer_ptr = self.inner.lock().unwrap().as_mut().addSlice(
            input.inner.lock().unwrap().as_mut(),
            &start_dims,
            &size_dims,
            &stride_dims,
        );
        Ok(SliceLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Slice))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addTopK`].
    pub fn add_topk(
        &mut self,
        input: &Tensor,
        op: TopKOperation,
        k: i32,
        axes: u32,
    ) -> Result<TopKLayer<'_>> {
        let layer_ptr = self.inner.lock().unwrap().as_mut().addTopK(
            input.inner.lock().unwrap().as_mut(),
            op.into(),
            k,
            axes,
        );
        Ok(TopKLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::TopK))?
        }))
    }
    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addResize`].
    pub fn add_resize(&mut self, input: &mut Tensor) -> Result<ResizeLayer<'_>> {
        let layer_ptr = self
            .inner
            .lock()
            .unwrap()
            .as_mut()
            .addResize(input.inner.lock().unwrap().as_mut());
        Ok(ResizeLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Resize))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addGather`].
    pub fn add_gather(
        &mut self,
        data: &mut Tensor,
        indices: &mut Tensor,
        axis: i32,
    ) -> Result<GatherLayer<'_>> {
        let layer_ptr = self.inner.lock().unwrap().as_mut().addGather(
            data.inner.lock().unwrap().as_mut(),
            indices.inner.lock().unwrap().as_mut(),
            axis,
        );
        Ok(GatherLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Gather))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addScatter`].
    pub fn add_scatter(
        &mut self,
        data: &mut Tensor,
        indices: &mut Tensor,
        updates: &mut Tensor,
        mode: trtx_sys::nvinfer1::ScatterMode,
    ) -> Result<ScatterLayer<'_>> {
        let layer_ptr = self.inner.lock().unwrap().as_mut().addScatter(
            data.inner.lock().unwrap().as_mut(),
            indices.inner.lock().unwrap().as_mut(),
            updates.inner.lock().unwrap().as_mut(),
            mode,
        );
        Ok(ScatterLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Scatter))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addQuantize`].
    pub fn add_quantize(
        &mut self,
        input: &mut Tensor,
        scale: &mut Tensor,
        output_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<QuantizeLayer<'_>> {
        let layer_ptr = self.inner.lock().unwrap().as_mut().addQuantize(
            input.inner.lock().unwrap().as_mut(),
            scale.inner.lock().unwrap().as_mut(),
            output_type,
        );
        Ok(QuantizeLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Quantize))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addDequantize`].
    pub fn add_dequantize(
        &mut self,
        input: &Tensor,
        scale: &Tensor,
        output_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<DequantizeLayer<'_>> {
        let layer_ptr = self.inner.lock().unwrap().as_mut().addDequantize(
            input.inner.lock().unwrap().as_mut(),
            scale.inner.lock().unwrap().as_mut(),
            output_type,
        );
        Ok(DequantizeLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Dequantize))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addSelect`].
    pub fn add_select(
        &mut self,
        condition: &Tensor,
        then_input: &Tensor,
        else_input: &Tensor,
    ) -> Result<SelectLayer<'_>> {
        let layer_ptr = self.inner.lock().unwrap().as_mut().addSelect(
            condition.inner.lock().unwrap().as_mut(),
            then_input.inner.lock().unwrap().as_mut(),
            else_input.inner.lock().unwrap().as_mut(),
        );
        Ok(SelectLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Select))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addPaddingNd`].
    pub fn add_padding(
        &self,
        input: &Tensor,
        pre_padding: &[i32],
        post_padding: &[i32],
    ) -> Result<PaddingLayer<'_>> {
        if pre_padding.len() != post_padding.len() {
            return Err(Error::Runtime(
                "pre_padding and post_padding must have the same length".to_string(),
            ));
        }
        let pre_i64: Vec<i64> = pre_padding.iter().map(|&d| d as i64).collect();
        let post_i64: Vec<i64> = post_padding.iter().map(|&d| d as i64).collect();
        let pre_dims = trtx_sys::Dims::from_slice(&pre_i64);
        let post_dims = trtx_sys::Dims::from_slice(&post_i64);
        let layer_ptr = self.inner.lock().unwrap().as_mut().addPaddingNd(
            input.inner.lock().unwrap().as_mut(),
            &pre_dims,
            &post_dims,
        );
        Ok(PaddingLayer::from_ptr(unsafe {
            layer_ptr
                .as_mut()
                .ok_or(Error::LayerCreationFailed(LayerTypeKind::Padding))?
        }))
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addAssertion`].
    pub fn add_assertion(&mut self, condition: &mut Tensor, message: &str) -> Result<()> {
        let message_cstr = std::ffi::CString::new(message)?;
        let layer_ptr = unsafe {
            self.inner.lock().unwrap().as_mut().addAssertion(
                condition.inner.lock().unwrap().as_mut(),
                message_cstr.as_ptr(),
            )
        };
        unsafe { layer_ptr.as_mut() }
            .ok_or(Error::LayerCreationFailed(LayerTypeKind::Assertion))?;
        Ok(())
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addLoop`].
    pub fn add_loop(&mut self) -> Result<Loop<'_>> {
        let loop_ptr = self.inner.lock().unwrap().as_mut().addLoop();
        let loop_ptr = unsafe { loop_ptr.as_mut() }
            .ok_or_else(|| Error::Runtime("Failed to add loop".to_string()))?;
        Ok(Loop {
            _inner: unsafe { Pin::new_unchecked(loop_ptr).into() },
        })
    }

    /// See [`trtx_sys::nvinfer1::INetworkDefinition::addIfConditional`].
    pub fn add_if_conditional(&mut self) -> Result<IfConditional<'_>> {
        let if_ptr = self.inner.lock().unwrap().as_mut().addIfConditional();
        Ok(IfConditional {
            _inner: unsafe {
                Pin::new_unchecked(
                    if_ptr
                        .as_mut()
                        .ok_or(Error::Runtime("Failed to add if conditional".to_string()))?,
                )
            }
            .into(),
        })
    }
}

// Network is owned by the builder; we hold a reference, so no deletion in Drop.
impl Drop for NetworkDefinition<'_> {
    fn drop(&mut self) {
        unsafe { ptr::drop_in_place(self.inner.get_mut().unwrap()) }
    }
}
