//! Real TensorRT network implementation
//! No #[cfg] - this module is only compiled when mock feature is disabled

use trtx_sys::nvinfer1::ScaleMode;

use crate::error::{Error, Result};
use crate::network::*;

/// Macro to implement Layer trait for real TensorRT types
macro_rules! impl_layer_real {
    ($name:ident, $trt_type:path) => {
        impl Layer for $name {
            fn get_output(&self, index: i32) -> Result<Tensor> {
                if self.inner.is_null() {
                    return Err(Error::Runtime("Invalid layer".to_string()));
                }
                let tensor_ptr = unsafe {
                    let layer_ref = &mut *(self.inner as *mut $trt_type);
                    layer_ref.as_ref().getOutput(index)
                };
                if tensor_ptr.is_null() {
                    return Err(Error::Runtime("Failed to get output tensor".to_string()));
                }
                Ok(Tensor {
                    inner: tensor_ptr as *mut _,
                })
            }
            fn as_ptr(&self) -> *mut std::ffi::c_void {
                self.inner
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

impl ShuffleLayer {
    pub fn set_reshape_dimensions(&mut self, dims: &[i32]) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid shuffle layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IShuffleLayer,
            >(self.inner);
            let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
            let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
            layer_pin.as_mut().setReshapeDimensions(&dims_obj);
        }
        Ok(())
    }
}

impl ResizeLayer {
    pub fn set_output_dimensions(&mut self, dims: &[i32]) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid resize layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IResizeLayer,
            >(self.inner);
            let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
            let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
            layer_pin.as_mut().setOutputDimensions(&dims_obj);
        }
        Ok(())
    }
    pub fn set_resize_mode(&mut self, mode: trtx_sys::ResizeMode) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid resize layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IResizeLayer,
            >(self.inner);
            layer_pin.as_mut().setResizeMode(mode);
        }
        Ok(())
    }
}

impl GatherLayer {
    pub fn set_gather_mode(&mut self, mode: trtx_sys::nvinfer1::GatherMode) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid gather layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IGatherLayer,
            >(self.inner);
            layer_pin.as_mut().setMode(mode);
        }
        Ok(())
    }
}

impl ScatterLayer {
    pub fn set_scatter_mode(&mut self, mode: trtx_sys::nvinfer1::ScatterMode) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid scatter layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IScatterLayer,
            >(self.inner);
            layer_pin.as_mut().setMode(mode);
        }
        Ok(())
    }
    pub fn set_axis(&mut self, axis: i32) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid scatter layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IScatterLayer,
            >(self.inner);
            layer_pin.as_mut().setAxis(axis);
        }
        Ok(())
    }
}

impl ConvolutionLayer {
    pub fn set_stride(&mut self, stride: &[i32; 2]) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid convolution layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IConvolutionLayer,
            >(self.inner);
            let dims_i64: Vec<i64> = stride.iter().map(|&s| s as i64).collect();
            let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
            layer_pin.as_mut().setStrideNd(&dims_obj);
        }
        Ok(())
    }
    pub fn set_padding(&mut self, padding: &[i32; 2]) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid convolution layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IConvolutionLayer,
            >(self.inner);
            let dims_i64: Vec<i64> = padding.iter().map(|&p| p as i64).collect();
            let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
            layer_pin.as_mut().setPaddingNd(&dims_obj);
        }
        Ok(())
    }
    pub fn set_dilation(&mut self, dilation: &[i32; 2]) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid convolution layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IConvolutionLayer,
            >(self.inner);
            let dims_i64: Vec<i64> = dilation.iter().map(|&d| d as i64).collect();
            let dims_obj = trtx_sys::Dims::from_slice(&dims_i64);
            layer_pin.as_mut().setDilationNd(&dims_obj);
        }
        Ok(())
    }
    pub fn set_num_groups(&mut self, num_groups: i32) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid convolution layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IConvolutionLayer,
            >(self.inner);
            layer_pin.as_mut().setNbGroups(num_groups as i64);
        }
        Ok(())
    }
}

impl ConcatenationLayer {
    pub fn set_axis(&mut self, axis: i32) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid concatenation layer".to_string()));
        }
        unsafe {
            let mut layer_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::IConcatenationLayer,
            >(self.inner);
            layer_pin.as_mut().setAxis(axis);
        }
        Ok(())
    }
}

impl Tensor {
    pub fn name(&self) -> Result<String> {
        let name_ptr = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ITensor>(self.inner)
                .getName()
        };
        if name_ptr.is_null() {
            return Err(Error::Runtime("Failed to get tensor name".to_string()));
        }
        unsafe { Ok(std::ffi::CStr::from_ptr(name_ptr).to_str()?.to_string()) }
    }

    pub fn set_name(&mut self, name: &str) -> Result<()> {
        let name_cstr = std::ffi::CString::new(name)?;
        unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ITensor>(self.inner)
                .setName(name_cstr.as_ptr());
        }
        Ok(())
    }

    pub fn dimensions(&self) -> Result<Vec<i32>> {
        let mut dims = [0i32; 8];
        let mut nb_dims = 0i32;
        let result =
            unsafe { trtx_sys::tensor_get_dimensions(self.inner, dims.as_mut_ptr(), &mut nb_dims) };
        if result.is_null() {
            return Err(Error::Runtime(
                "Failed to get tensor dimensions".to_string(),
            ));
        }
        if nb_dims < 0 {
            return Err(Error::Runtime(
                "Tensor dimensions not set (nbDims = -1)".to_string(),
            ));
        }
        Ok(dims[..nb_dims as usize].to_vec())
    }

    pub fn get_type(&self) -> Result<i32> {
        let data_type = unsafe { trtx_sys::tensor_get_type(self.inner) };
        Ok(data_type)
    }
}

impl NetworkDefinition {
    pub(crate) fn from_ptr(ptr: *mut std::ffi::c_void) -> Self {
        NetworkDefinition { inner: ptr }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.inner
    }

    pub fn add_input(
        &mut self,
        name: &str,
        data_type: trtx_sys::nvinfer1::DataType,
        dims: &[i32],
    ) -> Result<Tensor> {
        let name_cstr = std::ffi::CString::new(name)?;
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let dims_struct = trtx_sys::Dims::from_slice(&dims_i64);
        let network_ref =
            unsafe { &mut *(self.inner as *mut trtx_sys::nvinfer1::INetworkDefinition) };
        let mut network_pin = unsafe { std::pin::Pin::new_unchecked(network_ref) };
        let tensor_ptr = unsafe {
            network_pin
                .as_mut()
                .addInput(name_cstr.as_ptr(), data_type, &dims_struct)
        };
        if tensor_ptr.is_null() {
            return Err(Error::Runtime(format!("Failed to add input: {}", name)));
        }
        Ok(Tensor {
            inner: tensor_ptr as *mut _,
        })
    }

    pub fn mark_output(&mut self, tensor: &Tensor) -> Result<()> {
        unsafe {
            let tensor_ref = &mut *(tensor.inner as *mut trtx_sys::nvinfer1::ITensor);
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(
                self.inner,
            )
            .markOutput(std::pin::Pin::new_unchecked(tensor_ref));
        }
        Ok(())
    }

    pub fn get_nb_inputs(&self) -> i32 {
        unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(
                self.inner,
            )
            .getNbInputs()
        }
    }

    pub fn get_nb_outputs(&self) -> i32 {
        unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(
                self.inner,
            )
            .getNbOutputs()
        }
    }

    pub fn get_input(&self, index: i32) -> Result<Tensor> {
        let tensor_ptr = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(
                self.inner,
            )
            .getInput(index)
        };
        if tensor_ptr.is_null() {
            return Err(Error::Runtime(format!(
                "Failed to get input at index {}",
                index
            )));
        }
        Ok(Tensor {
            inner: tensor_ptr as *mut _,
        })
    }

    pub fn get_output(&self, index: i32) -> Result<Tensor> {
        let tensor_ptr = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(
                self.inner,
            )
            .getOutput(index)
        };
        if tensor_ptr.is_null() {
            return Err(Error::Runtime(format!(
                "Failed to get output at index {}",
                index
            )));
        }
        Ok(Tensor {
            inner: tensor_ptr as *mut _,
        })
    }

    pub fn add_activation(
        &mut self,
        input: &Tensor,
        activation_type: trtx_sys::nvinfer1::ActivationType,
    ) -> Result<ActivationLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin
                .as_mut()
                .addActivation(std::pin::Pin::new_unchecked(input_ref), activation_type);
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add activation layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(ActivationLayer::from_ptr(layer_ptr))
    }

    pub fn add_unary(
        &mut self,
        input: &Tensor,
        op: trtx_sys::nvinfer1::UnaryOperation,
    ) -> Result<UnaryLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin
                .as_mut()
                .addUnary(std::pin::Pin::new_unchecked(input_ref), op);
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add unary layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(UnaryLayer::from_ptr(layer_ptr))
    }

    pub fn add_identity(&mut self, input: &Tensor) -> Result<IdentityLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin
                .as_mut()
                .addIdentity(std::pin::Pin::new_unchecked(input_ref));
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add identity layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(IdentityLayer::from_ptr(layer_ptr))
    }

    pub fn add_cast(
        &mut self,
        input: &Tensor,
        to_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<CastLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin
                .as_mut()
                .addCast(std::pin::Pin::new_unchecked(input_ref), to_type);
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add cast layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(CastLayer::from_ptr(layer_ptr))
    }

    pub fn add_elementwise(
        &mut self,
        input1: &Tensor,
        input2: &Tensor,
        op: trtx_sys::nvinfer1::ElementWiseOperation,
    ) -> Result<ElementWiseLayer> {
        let layer_ptr = unsafe {
            let input1_ref = &mut *(input1.inner as *mut trtx_sys::nvinfer1::ITensor);
            let input2_ref = &mut *(input2.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin.as_mut().addElementWise(
                std::pin::Pin::new_unchecked(input1_ref),
                std::pin::Pin::new_unchecked(input2_ref),
                op,
            );
            if layer_ptr.is_null() {
                return Err(Error::Runtime(
                    "Failed to add elementwise layer".to_string(),
                ));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(ElementWiseLayer::from_ptr(layer_ptr))
    }

    pub fn add_pooling(
        &mut self,
        input: &Tensor,
        pooling_type: trtx_sys::nvinfer1::PoolingType,
        window_size: &[i32; 2],
    ) -> Result<PoolingLayer> {
        let window_dims = trtx_sys::Dims::new_2d(window_size[0] as i64, window_size[1] as i64);
        let network_ref =
            unsafe { &mut *(self.inner as *mut trtx_sys::nvinfer1::INetworkDefinition) };
        let mut network_pin = unsafe { std::pin::Pin::new_unchecked(network_ref) };
        let input_ref = unsafe { &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor) };
        let mut input_pin = unsafe { std::pin::Pin::new_unchecked(input_ref) };
        let layer_ptr =
            network_pin
                .as_mut()
                .addPoolingNd(input_pin.as_mut(), pooling_type, &window_dims);
        if layer_ptr.is_null() {
            return Err(Error::Runtime("Failed to add pooling layer".to_string()));
        }
        Ok(PoolingLayer::from_ptr(layer_ptr as *mut _))
    }

    pub fn add_shuffle(&mut self, input: &Tensor) -> Result<ShuffleLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin
                .as_mut()
                .addShuffle(std::pin::Pin::new_unchecked(input_ref));
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add shuffle layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(ShuffleLayer::from_ptr(layer_ptr))
    }

    pub fn add_matrix_multiply(
        &mut self,
        input0: &Tensor,
        op0: i32,
        input1: &Tensor,
        op1: i32,
    ) -> Result<MatrixMultiplyLayer> {
        let layer_ptr = unsafe {
            let input0_ref = &mut *(input0.inner as *mut trtx_sys::nvinfer1::ITensor);
            let input1_ref = &mut *(input1.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin.as_mut().addMatrixMultiply(
                std::pin::Pin::new_unchecked(input0_ref),
                std::mem::transmute::<i32, trtx_sys::nvinfer1::MatrixOperation>(op0),
                std::pin::Pin::new_unchecked(input1_ref),
                std::mem::transmute::<i32, trtx_sys::nvinfer1::MatrixOperation>(op1),
            );
            if layer_ptr.is_null() {
                return Err(Error::Runtime(
                    "Failed to add matrix multiply layer".to_string(),
                ));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(MatrixMultiplyLayer::from_ptr(layer_ptr))
    }

    pub fn add_convolution(
        &mut self,
        input: &Tensor,
        nb_output_maps: i32,
        kernel_size: &[i32; 2],
        kernel_weights: &[u8],
        bias_weights: Option<&[u8]>,
    ) -> Result<ConvolutionLayer> {
        let weight_count = (kernel_weights.len() / 4) as i64;
        let bias_count = if bias_weights.is_some() {
            nb_output_maps as i64
        } else {
            0
        };
        let bias_ptr = bias_weights
            .map(|b| b.as_ptr() as *const std::ffi::c_void)
            .unwrap_or(std::ptr::null());
        let kernel_dims = trtx_sys::Dims::new_2d(kernel_size[0] as i64, kernel_size[1] as i64);
        let kernel_w = trtx_sys::nvinfer1::Weights::new_float(
            kernel_weights.as_ptr() as *const std::ffi::c_void,
            weight_count,
        );
        let bias_w = trtx_sys::nvinfer1::Weights::new_float(bias_ptr, bias_count);
        let layer_ptr = unsafe {
            let network_ptr = self.inner as *mut trtx_sys::nvinfer1::INetworkDefinition;
            let mut network_pin = std::pin::Pin::new_unchecked(&mut *network_ptr);
            let input_ptr = input.inner as *mut trtx_sys::nvinfer1::ITensor;
            let input_ref = std::pin::Pin::new_unchecked(&mut *input_ptr);
            network_pin.as_mut().addConvolutionNd(
                input_ref,
                nb_output_maps as i64,
                &kernel_dims,
                kernel_w,
                bias_w,
            ) as *mut std::ffi::c_void
        };
        if layer_ptr.is_null() {
            return Err(Error::Runtime(
                "Failed to add convolution layer".to_string(),
            ));
        }
        Ok(ConvolutionLayer::from_ptr(layer_ptr))
    }

    pub fn add_deconvolution(
        &mut self,
        input: &Tensor,
        nb_output_maps: i32,
        kernel_size: &[i32; 2],
        kernel_weights: &[u8],
        bias_weights: Option<&[u8]>,
    ) -> Result<DeconvolutionLayer> {
        let input_dims = input.dimensions()?;
        let input_channels = if input_dims.len() >= 4 {
            input_dims[1] as i64
        } else if input_dims.len() >= 3 {
            input_dims[0] as i64
        } else {
            return Err(Error::InvalidArgument(format!(
                "Invalid input dimensions for deconvolution: {:?}",
                input_dims
            )));
        };
        let weight_count =
            nb_output_maps as i64 * input_channels * kernel_size[0] as i64 * kernel_size[1] as i64;
        let bias_count = if bias_weights.is_some() {
            nb_output_maps as i64
        } else {
            0
        };
        let bias_ptr = bias_weights
            .map(|b| b.as_ptr() as *const std::ffi::c_void)
            .unwrap_or(std::ptr::null());
        let kernel_dims = trtx_sys::Dims::new_2d(kernel_size[0] as i64, kernel_size[1] as i64);
        let kernel_w = trtx_sys::nvinfer1::Weights::new_float(
            kernel_weights.as_ptr() as *const std::ffi::c_void,
            weight_count,
        );
        let bias_w = trtx_sys::nvinfer1::Weights::new_float(bias_ptr, bias_count);
        let layer_ptr = unsafe {
            let network_ptr = self.inner as *mut trtx_sys::nvinfer1::INetworkDefinition;
            let mut network_pin = std::pin::Pin::new_unchecked(&mut *network_ptr);
            let input_ptr = input.inner as *mut trtx_sys::nvinfer1::ITensor;
            let input_ref = std::pin::Pin::new_unchecked(&mut *input_ptr);
            network_pin.as_mut().addDeconvolutionNd(
                input_ref,
                nb_output_maps as i64,
                kernel_dims,
                kernel_w,
                bias_w,
            ) as *mut std::ffi::c_void
        };
        if layer_ptr.is_null() {
            return Err(Error::Runtime(
                "Failed to add deconvolution layer".to_string(),
            ));
        }
        Ok(DeconvolutionLayer::from_ptr(layer_ptr))
    }

    pub fn add_concatenation(&mut self, inputs: &[&Tensor]) -> Result<ConcatenationLayer> {
        let mut input_ptrs: Vec<*mut std::ffi::c_void> = inputs.iter().map(|t| t.inner).collect();
        let layer_ptr = unsafe {
            trtx_sys::network_add_concatenation(
                self.inner,
                input_ptrs.as_mut_ptr(),
                inputs.len() as i32,
            )
        };
        if layer_ptr.is_null() {
            return Err(Error::Runtime(
                "Failed to add concatenation layer".to_string(),
            ));
        }
        Ok(ConcatenationLayer::from_ptr(layer_ptr))
    }

    pub fn add_constant(
        &mut self,
        dims: &[i32],
        weights: &[u8],
        data_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<ConstantLayer> {
        use trtx_sys::nvinfer1::DataType;
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
                    "Unsupported data type: {}",
                    crate::datatype_name(&data_type)
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
            data_type,
            weights.as_ptr() as *const std::ffi::c_void,
            element_count,
        );
        let layer_ptr = unsafe {
            let network_ptr = self.inner as *mut trtx_sys::nvinfer1::INetworkDefinition;
            let mut network_pin = std::pin::Pin::new_unchecked(&mut *network_ptr);
            network_pin
                .as_mut()
                .addConstant(&dims_struct, weights_struct) as *mut std::ffi::c_void
        };
        if layer_ptr.is_null() {
            return Err(Error::Runtime("Failed to add constant tensor".to_string()));
        }
        Ok(ConstantLayer::from_ptr(layer_ptr))
    }

    pub fn add_softmax(&mut self, input: &Tensor, axes: u32) -> Result<SoftMaxLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin
                .as_mut()
                .addSoftMax(std::pin::Pin::new_unchecked(input_ref));
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add softmax layer".to_string()));
            }
            let mut layer_pin = std::pin::Pin::new_unchecked(&mut *layer_ptr);
            layer_pin.as_mut().setAxes(axes);
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(SoftMaxLayer::from_ptr(layer_ptr))
    }

    pub fn add_scale(
        &mut self,
        input: &Tensor,
        mode: ScaleMode,
        shift: &[u8],
        scale: &[u8],
        power: &[u8],
    ) -> Result<ScaleLayer> {
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
        let layer_ptr = unsafe {
            let network_ptr = self.inner as *mut trtx_sys::nvinfer1::INetworkDefinition;
            let mut network_pin = std::pin::Pin::new_unchecked(&mut *network_ptr);
            let input_ptr = input.inner as *mut trtx_sys::nvinfer1::ITensor;
            let input_ref = std::pin::Pin::new_unchecked(&mut *input_ptr);
            network_pin
                .as_mut()
                .addScale(input_ref, mode, shift_w, scale_w, power_w)
                as *mut std::ffi::c_void
        };
        if layer_ptr.is_null() {
            return Err(Error::Runtime("Failed to add scale layer".to_string()));
        }
        Ok(ScaleLayer::from_ptr(layer_ptr))
    }

    pub fn add_reduce(
        &mut self,
        input: &Tensor,
        op: trtx_sys::nvinfer1::ReduceOperation,
        axes: u32,
        keep_dims: bool,
    ) -> Result<ReduceLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin.as_mut().addReduce(
                std::pin::Pin::new_unchecked(input_ref),
                op,
                axes,
                keep_dims,
            );
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add reduce layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(ReduceLayer::from_ptr(layer_ptr))
    }

    pub fn add_cumulative(
        &mut self,
        input: &Tensor,
        axis: i32,
        op: trtx_sys::nvinfer1::CumulativeOperation,
        exclusive: bool,
        reverse: bool,
    ) -> Result<CumulativeLayer> {
        let axis_bytes = axis.to_le_bytes();
        let axis_constant =
            self.add_constant(&[], &axis_bytes, trtx_sys::nvinfer1::DataType::kINT32)?;
        let axis_tensor = axis_constant.get_output(0)?;
        self.add_cumulative_with_axis_tensor(input, &axis_tensor, op, exclusive, reverse)
    }

    pub fn add_cumulative_with_axis_tensor(
        &mut self,
        input: &Tensor,
        axis_tensor: &Tensor,
        op: trtx_sys::nvinfer1::CumulativeOperation,
        exclusive: bool,
        reverse: bool,
    ) -> Result<CumulativeLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let axis_ref = &mut *(axis_tensor.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin.as_mut().addCumulative(
                std::pin::Pin::new_unchecked(input_ref),
                std::pin::Pin::new_unchecked(axis_ref),
                op,
                exclusive,
                reverse,
            );
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add cumulative layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(CumulativeLayer::from_ptr(layer_ptr))
    }

    pub fn add_slice(
        &mut self,
        input: &Tensor,
        start: &[i32],
        size: &[i32],
        stride: &[i32],
    ) -> Result<SliceLayer> {
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
        let network_ref =
            unsafe { &mut *(self.inner as *mut trtx_sys::nvinfer1::INetworkDefinition) };
        let mut network_pin = unsafe { std::pin::Pin::new_unchecked(network_ref) };
        let input_ref = unsafe { &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor) };
        let mut input_pin = unsafe { std::pin::Pin::new_unchecked(input_ref) };
        let layer_ptr = network_pin.as_mut().addSlice(
            input_pin.as_mut(),
            &start_dims,
            &size_dims,
            &stride_dims,
        );
        if layer_ptr.is_null() {
            return Err(Error::Runtime("Failed to add slice layer".to_string()));
        }
        Ok(SliceLayer::from_ptr(layer_ptr as *mut _))
    }

    pub fn add_resize(&mut self, input: &Tensor) -> Result<ResizeLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin
                .as_mut()
                .addResize(std::pin::Pin::new_unchecked(input_ref));
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add resize layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(ResizeLayer::from_ptr(layer_ptr))
    }

    pub fn add_topk(&mut self, input: &Tensor, op: i32, k: i32, axes: u32) -> Result<TopKLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin.as_mut().addTopK(
                std::pin::Pin::new_unchecked(input_ref),
                std::mem::transmute::<i32, trtx_sys::nvinfer1::TopKOperation>(op),
                k,
                axes,
            );
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add topk layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(TopKLayer::from_ptr(layer_ptr))
    }

    pub fn add_gather(
        &mut self,
        data: &Tensor,
        indices: &Tensor,
        axis: i32,
    ) -> Result<GatherLayer> {
        let layer_ptr = unsafe {
            let data_ref = &mut *(data.inner as *mut trtx_sys::nvinfer1::ITensor);
            let indices_ref = &mut *(indices.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin.as_mut().addGather(
                std::pin::Pin::new_unchecked(data_ref),
                std::pin::Pin::new_unchecked(indices_ref),
                axis,
            );
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add gather layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(GatherLayer::from_ptr(layer_ptr))
    }

    pub fn add_scatter(
        &mut self,
        data: &Tensor,
        indices: &Tensor,
        updates: &Tensor,
        mode: trtx_sys::nvinfer1::ScatterMode,
    ) -> Result<ScatterLayer> {
        let layer_ptr = unsafe {
            let data_ref = &mut *(data.inner as *mut trtx_sys::nvinfer1::ITensor);
            let indices_ref = &mut *(indices.inner as *mut trtx_sys::nvinfer1::ITensor);
            let updates_ref = &mut *(updates.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin.as_mut().addScatter(
                std::pin::Pin::new_unchecked(data_ref),
                std::pin::Pin::new_unchecked(indices_ref),
                std::pin::Pin::new_unchecked(updates_ref),
                mode,
            );
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add scatter layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(ScatterLayer::from_ptr(layer_ptr))
    }

    pub fn add_quantize(
        &mut self,
        input: &Tensor,
        scale: &Tensor,
        output_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<QuantizeLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let scale_ref = &mut *(scale.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin.as_mut().addQuantize(
                std::pin::Pin::new_unchecked(input_ref),
                std::pin::Pin::new_unchecked(scale_ref),
                output_type,
            );
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add quantize layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(QuantizeLayer::from_ptr(layer_ptr))
    }

    pub fn add_dequantize(
        &mut self,
        input: &Tensor,
        scale: &Tensor,
        output_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<DequantizeLayer> {
        let layer_ptr = unsafe {
            let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let scale_ref = &mut *(scale.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin.as_mut().addDequantize(
                std::pin::Pin::new_unchecked(input_ref),
                std::pin::Pin::new_unchecked(scale_ref),
                output_type,
            );
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add dequantize layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(DequantizeLayer::from_ptr(layer_ptr))
    }

    pub fn add_select(
        &mut self,
        condition: &Tensor,
        then_input: &Tensor,
        else_input: &Tensor,
    ) -> Result<SelectLayer> {
        let layer_ptr = unsafe {
            let condition_ref = &mut *(condition.inner as *mut trtx_sys::nvinfer1::ITensor);
            let then_ref = &mut *(then_input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let else_ref = &mut *(else_input.inner as *mut trtx_sys::nvinfer1::ITensor);
            let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                trtx_sys::nvinfer1::INetworkDefinition,
            >(self.inner);
            let layer_ptr = network_pin.as_mut().addSelect(
                std::pin::Pin::new_unchecked(condition_ref),
                std::pin::Pin::new_unchecked(then_ref),
                std::pin::Pin::new_unchecked(else_ref),
            );
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add select layer".to_string()));
            }
            layer_ptr as *mut std::ffi::c_void
        };
        Ok(SelectLayer::from_ptr(layer_ptr))
    }

    pub fn add_padding(
        &mut self,
        input: &Tensor,
        pre_padding: &[i32],
        post_padding: &[i32],
    ) -> Result<PaddingLayer> {
        if pre_padding.len() != post_padding.len() {
            return Err(Error::Runtime(
                "pre_padding and post_padding must have the same length".to_string(),
            ));
        }
        let pre_i64: Vec<i64> = pre_padding.iter().map(|&d| d as i64).collect();
        let post_i64: Vec<i64> = post_padding.iter().map(|&d| d as i64).collect();
        let pre_dims = trtx_sys::Dims::from_slice(&pre_i64);
        let post_dims = trtx_sys::Dims::from_slice(&post_i64);
        let network_ref =
            unsafe { &mut *(self.inner as *mut trtx_sys::nvinfer1::INetworkDefinition) };
        let mut network_pin = unsafe { std::pin::Pin::new_unchecked(network_ref) };
        let input_ref = unsafe { &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor) };
        let mut input_pin = unsafe { std::pin::Pin::new_unchecked(input_ref) };
        let layer_ptr =
            network_pin
                .as_mut()
                .addPaddingNd(input_pin.as_mut(), &pre_dims, &post_dims);
        if layer_ptr.is_null() {
            return Err(Error::Runtime("Failed to add padding layer".to_string()));
        }
        Ok(PaddingLayer::from_ptr(layer_ptr as *mut _))
    }

    pub fn add_assertion(&mut self, condition: &Tensor, message: &str) -> Result<()> {
        let message_cstr = std::ffi::CString::new(message)?;
        let layer_ptr = unsafe {
            trtx_sys::network_add_assertion(self.inner, condition.inner, message_cstr.as_ptr())
        };
        if layer_ptr.is_null() {
            return Err(Error::Runtime("Failed to add assertion layer".to_string()));
        }
        Ok(())
    }

    pub fn add_loop(&mut self) -> Result<*mut std::ffi::c_void> {
        let loop_ptr = unsafe { trtx_sys::network_add_loop(self.inner) };
        if loop_ptr.is_null() {
            return Err(Error::Runtime("Failed to add loop".to_string()));
        }
        Ok(loop_ptr)
    }

    pub fn add_if_conditional(&mut self) -> Result<*mut std::ffi::c_void> {
        let if_ptr = unsafe { trtx_sys::network_add_if_conditional(self.inner) };
        if if_ptr.is_null() {
            return Err(Error::Runtime("Failed to add if conditional".to_string()));
        }
        Ok(if_ptr)
    }
}

impl Drop for NetworkDefinition {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_sys::delete_network(self.inner);
            }
        }
    }
}

unsafe impl Send for NetworkDefinition {}
