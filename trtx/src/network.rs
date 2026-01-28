//! Network definition for building TensorRT engines

use crate::error::{Error, Result};

/// Tensor handle (opaque pointer)
pub struct Tensor {
    inner: *mut std::ffi::c_void,
}

//==============================================================================
// Layer Types
//==============================================================================

/// Base trait for all layer types
pub trait Layer {
    /// Get the output tensor at the specified index
    fn get_output(&self, index: i32) -> Result<Tensor>;

    /// Get the raw layer pointer (for internal use)
    fn as_ptr(&self) -> *mut std::ffi::c_void;
}

/// Macro to define layer wrapper types
macro_rules! define_layer {
    ($name:ident, $trt_type:path) => {
        pub struct $name {
            inner: *mut std::ffi::c_void,
        }

        impl $name {
            pub(crate) fn from_ptr(ptr: *mut std::ffi::c_void) -> Self {
                Self { inner: ptr }
            }
        }

        impl Layer for $name {
            fn get_output(&self, index: i32) -> Result<Tensor> {
                #[cfg(not(feature = "mock"))]
                {
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
                #[cfg(feature = "mock")]
                {
                    Ok(Tensor {
                        inner: std::ptr::null_mut(),
                    })
                }
            }

            fn as_ptr(&self) -> *mut std::ffi::c_void {
                self.inner
            }
        }
    };
}

// Define all layer types
define_layer!(ShuffleLayer, trtx_sys::nvinfer1::IShuffleLayer);
define_layer!(ActivationLayer, trtx_sys::nvinfer1::IActivationLayer);
define_layer!(ElementWiseLayer, trtx_sys::nvinfer1::IElementWiseLayer);
define_layer!(ResizeLayer, trtx_sys::nvinfer1::IResizeLayer);
define_layer!(TopKLayer, trtx_sys::nvinfer1::ITopKLayer);
define_layer!(GatherLayer, trtx_sys::nvinfer1::IGatherLayer);
define_layer!(SelectLayer, trtx_sys::nvinfer1::ISelectLayer);
define_layer!(
    MatrixMultiplyLayer,
    trtx_sys::nvinfer1::IMatrixMultiplyLayer
);
define_layer!(SoftMaxLayer, trtx_sys::nvinfer1::ISoftMaxLayer);
define_layer!(ReduceLayer, trtx_sys::nvinfer1::IReduceLayer);
define_layer!(PoolingLayer, trtx_sys::nvinfer1::IPoolingLayer);
define_layer!(ConvolutionLayer, trtx_sys::nvinfer1::IConvolutionLayer);
define_layer!(ConstantLayer, trtx_sys::nvinfer1::IConstantLayer);
define_layer!(ConcatenationLayer, trtx_sys::nvinfer1::IConcatenationLayer);
define_layer!(ScaleLayer, trtx_sys::nvinfer1::IScaleLayer);
define_layer!(SliceLayer, trtx_sys::nvinfer1::ISliceLayer);
define_layer!(UnaryLayer, trtx_sys::nvinfer1::IUnaryLayer);

impl Tensor {
    /// Get the tensor name
    pub fn name(&self) -> Result<String> {
        #[cfg(not(feature = "mock"))]
        {
            // Use autocxx Pin to call getName directly
            let name_ptr = unsafe {
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ITensor>(self.inner)
                    .getName()
            };
            if name_ptr.is_null() {
                return Err(Error::Runtime("Failed to get tensor name".to_string()));
            }
            unsafe { Ok(std::ffi::CStr::from_ptr(name_ptr).to_str()?.to_string()) }
        }
        #[cfg(feature = "mock")]
        {
            Ok("mock_tensor".to_string())
        }
    }

    /// Set the tensor name
    pub fn set_name(&mut self, name: &str) -> Result<()> {
        #[cfg(not(feature = "mock"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            // Use autocxx Pin to call setName directly
            unsafe {
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ITensor>(self.inner)
                    .setName(name_cstr.as_ptr());
            }
            Ok(())
        }
        #[cfg(feature = "mock")]
        {
            Ok(())
        }
    }

    /// Get tensor dimensions
    pub fn dimensions(&self) -> Result<Vec<i32>> {
        #[cfg(not(feature = "mock"))]
        {
            // Keep wrapper - getDimensions returns Dims which is complex to extract
            let mut dims = [0i32; 8]; // MAX_DIMS
            let mut nb_dims = 0i32;
            let result = unsafe {
                trtx_sys::tensor_get_dimensions(self.inner, dims.as_mut_ptr(), &mut nb_dims)
            };
            if result.is_null() {
                return Err(Error::Runtime(
                    "Failed to get tensor dimensions".to_string(),
                ));
            }
            Ok(dims[..nb_dims as usize].to_vec())
        }
        #[cfg(feature = "mock")]
        {
            Ok(vec![1, 3, 224, 224])
        }
    }

    /// Get tensor data type
    pub fn get_type(&self) -> Result<i32> {
        #[cfg(not(feature = "mock"))]
        {
            // Keep wrapper - DataType enum conversion is simpler through wrapper
            let data_type = unsafe { trtx_sys::tensor_get_type(self.inner) };
            Ok(data_type)
        }
        #[cfg(feature = "mock")]
        {
            Ok(0) // DataType::kFLOAT
        }
    }
}

/// Network definition for building TensorRT engines
pub struct NetworkDefinition {
    #[cfg(not(feature = "mock"))]
    inner: *mut std::ffi::c_void,
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxNetworkDefinition,
}

impl NetworkDefinition {
    /// Create a new NetworkDefinition from a raw pointer (internal use)
    #[cfg(not(feature = "mock"))]
    pub(crate) fn from_ptr(ptr: *mut std::ffi::c_void) -> Self {
        NetworkDefinition { inner: ptr }
    }

    #[cfg(feature = "mock")]
    pub(crate) fn from_ptr(ptr: *mut trtx_sys::TrtxNetworkDefinition) -> Self {
        NetworkDefinition { inner: ptr }
    }

    /// Get the raw pointer (for internal use)
    #[cfg(not(feature = "mock"))]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.inner
    }

    #[cfg(feature = "mock")]
    pub(crate) fn as_ptr(&self) -> *mut trtx_sys::TrtxNetworkDefinition {
        self.inner
    }

    #[cfg(feature = "mock")]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut trtx_sys::TrtxNetworkDefinition {
        self.inner
    }

    /// Add an input tensor to the network
    pub fn add_input(&mut self, name: &str, data_type: i32, dims: &[i32]) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            
            // Convert i32 dims to i64 and create Dims structure
            let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
            let dims_struct = trtx_sys::Dims::from_slice(&dims_i64);
            
            let tensor_ptr = unsafe {
                trtx_sys::network_add_input(
                    self.inner,
                    name_cstr.as_ptr(),
                    data_type,
                    &dims_struct as *const _,
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime(format!("Failed to add input: {}", name)));
            }
            Ok(Tensor { inner: tensor_ptr })
        }
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Mark a tensor as a network output
    pub fn mark_output(&mut self, tensor: &Tensor) -> Result<()> {
        #[cfg(not(feature = "mock"))]
        {
            unsafe {
                let tensor_ref = &mut *(tensor.inner as *mut trtx_sys::nvinfer1::ITensor);
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(
                    self.inner,
                )
                .markOutput(std::pin::Pin::new_unchecked(tensor_ref));
            }
            Ok(())
        }
        #[cfg(feature = "mock")]
        {
            Ok(())
        }
    }

    /// Get the number of inputs
    pub fn get_nb_inputs(&self) -> i32 {
        #[cfg(not(feature = "mock"))]
        unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(
                self.inner,
            )
            .getNbInputs()
        }
        #[cfg(feature = "mock")]
        {
            0
        }
    }

    /// Get the number of outputs
    pub fn get_nb_outputs(&self) -> i32 {
        #[cfg(not(feature = "mock"))]
        unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(
                self.inner,
            )
            .getNbOutputs()
        }
        #[cfg(feature = "mock")]
        {
            0
        }
    }

    /// Get an input tensor by index
    pub fn get_input(&self, index: i32) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
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
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Get an output tensor by index
    pub fn get_output(&self, index: i32) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
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
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Add an activation layer
    pub fn add_activation(
        &mut self,
        input: &Tensor,
        activation_type: i32,
    ) -> Result<ActivationLayer> {
        #[cfg(not(feature = "mock"))]
        {
            let layer_ptr = unsafe {
                let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
                let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                    trtx_sys::nvinfer1::INetworkDefinition,
                >(self.inner);

                let layer_ptr = network_pin.as_mut().addActivation(
                    std::pin::Pin::new_unchecked(input_ref),
                    std::mem::transmute::<i32, trtx_sys::nvinfer1::ActivationType>(activation_type),
                );

                if layer_ptr.is_null() {
                    return Err(Error::Runtime("Failed to add activation layer".to_string()));
                }

                layer_ptr as *mut std::ffi::c_void
            };

            Ok(ActivationLayer::from_ptr(layer_ptr))
        }
        #[cfg(feature = "mock")]
        {
            Ok(ActivationLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a unary operation layer
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `op` - Unary operation type (e.g., EXP, LOG, SQRT, ABS, NEG, CEIL, FLOOR, etc.)
    pub fn add_unary(&mut self, input: &Tensor, op: i32) -> Result<UnaryLayer> {
        #[cfg(not(feature = "mock"))]
        {
            let layer_ptr = unsafe {
                let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
                let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                    trtx_sys::nvinfer1::INetworkDefinition,
                >(self.inner);

                let layer_ptr = network_pin.as_mut().addUnary(
                    std::pin::Pin::new_unchecked(input_ref),
                    std::mem::transmute::<i32, trtx_sys::nvinfer1::UnaryOperation>(op),
                );

                if layer_ptr.is_null() {
                    return Err(Error::Runtime("Failed to add unary layer".to_string()));
                }

                layer_ptr as *mut std::ffi::c_void
            };

            Ok(UnaryLayer::from_ptr(layer_ptr))
        }
        #[cfg(feature = "mock")]
        {
            Ok(UnaryLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add an elementwise operation layer
    pub fn add_elementwise(
        &mut self,
        input1: &Tensor,
        input2: &Tensor,
        op: i32,
    ) -> Result<ElementWiseLayer> {
        #[cfg(not(feature = "mock"))]
        {
            let layer_ptr = unsafe {
                let input1_ref = &mut *(input1.inner as *mut trtx_sys::nvinfer1::ITensor);
                let input2_ref = &mut *(input2.inner as *mut trtx_sys::nvinfer1::ITensor);
                let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                    trtx_sys::nvinfer1::INetworkDefinition,
                >(self.inner);

                let layer_ptr = network_pin.as_mut().addElementWise(
                    std::pin::Pin::new_unchecked(input1_ref),
                    std::pin::Pin::new_unchecked(input2_ref),
                    std::mem::transmute::<i32, trtx_sys::nvinfer1::ElementWiseOperation>(op),
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
        #[cfg(feature = "mock")]
        {
            Ok(ElementWiseLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a pooling layer
    pub fn add_pooling(
        &mut self,
        input: &Tensor,
        pooling_type: i32,
        window_size: &[i32; 2],
    ) -> Result<PoolingLayer> {
        #[cfg(not(feature = "mock"))]
        {
            let layer_ptr = unsafe {
                trtx_sys::network_add_pooling(
                    self.inner,
                    input.inner,
                    pooling_type,
                    window_size.as_ptr(),
                )
            };
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add pooling layer".to_string()));
            }
            Ok(PoolingLayer::from_ptr(layer_ptr))
        }
        #[cfg(feature = "mock")]
        {
            Ok(PoolingLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a shuffle layer (for reshaping/transposing)
    pub fn add_shuffle(&mut self, input: &Tensor) -> Result<ShuffleLayer> {
        #[cfg(not(feature = "mock"))]
        {
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
        #[cfg(feature = "mock")]
        {
            Ok(ShuffleLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a matrix multiply layer
    ///
    /// # Arguments
    /// * `input0` - First input tensor
    /// * `op0` - Matrix operation for first input (0=NONE, 1=TRANSPOSE, 2=VECTOR)
    /// * `input1` - Second input tensor
    /// * `op1` - Matrix operation for second input (0=NONE, 1=TRANSPOSE, 2=VECTOR)
    ///
    /// # Matrix Operations
    /// - `0` (kNONE): Use matrix as-is
    /// - `1` (kTRANSPOSE): Transpose the matrix
    /// - `2` (kVECTOR): Treat as 1D vector
    pub fn add_matrix_multiply(
        &mut self,
        input0: &Tensor,
        op0: i32,
        input1: &Tensor,
        op1: i32,
    ) -> Result<MatrixMultiplyLayer> {
        #[cfg(not(feature = "mock"))]
        {
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
        #[cfg(feature = "mock")]
        {
            Ok(MatrixMultiplyLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a convolution layer
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `nb_output_maps` - Number of output feature maps
    /// * `kernel_size` - Kernel dimensions [height, width]
    /// * `kernel_weights` - Kernel weights as raw bytes (float32)
    ///   - Size: `nb_output_maps * input_channels * kernel_size[0] * kernel_size[1] * sizeof(f32)`
    ///   - Format: [output_channel, input_channel, kernel_height, kernel_width]
    /// * `bias_weights` - Optional bias weights as raw bytes (float32)
    ///   - Size: `nb_output_maps * sizeof(f32)` if provided
    ///
    /// # Note
    /// The caller must ensure the weight data has the correct size.
    /// Weight count is calculated based on the input tensor dimensions.
    pub fn add_convolution(
        &mut self,
        input: &Tensor,
        nb_output_maps: i32,
        kernel_size: &[i32; 2],
        kernel_weights: &[u8],
        bias_weights: Option<&[u8]>,
    ) -> Result<ConvolutionLayer> {
        #[cfg(not(feature = "mock"))]
        {
            // Calculate weight count based on input dimensions
            let input_dims = input.dimensions()?;
            let input_channels = if input_dims.len() >= 4 {
                // Format: [N, C, H, W] - channels at index 1
                input_dims[1] as i64
            } else if input_dims.len() >= 3 {
                // Format: [C, H, W] - channels at index 0
                input_dims[0] as i64
            } else {
                return Err(Error::InvalidArgument(format!(
                    "Invalid input dimensions for convolution: {:?}",
                    input_dims
                )));
            };

            // weight_count = nb_outputs * input_channels * kernel_h * kernel_w (in floats)
            let weight_count = nb_output_maps as i64 * input_channels * 
                               kernel_size[0] as i64 * kernel_size[1] as i64;

            let bias_count = if bias_weights.is_some() {
                nb_output_maps as i64
            } else {
                0
            };

            let bias_ptr = bias_weights
                .map(|b| b.as_ptr() as *const std::ffi::c_void)
                .unwrap_or(std::ptr::null());

            // Create Dims structure for kernel (i64 dimensions for Dims64)
            let kernel_dims = trtx_sys::Dims::new_2d(kernel_size[0] as i64, kernel_size[1] as i64);

            let layer_ptr = unsafe {
                trtx_sys::network_add_convolution(
                    self.inner,
                    input.inner,
                    nb_output_maps,
                    &kernel_dims as *const _,
                    kernel_weights.as_ptr() as *const std::ffi::c_void,
                    weight_count,
                    bias_ptr,
                    bias_count,
                )
            };
            if layer_ptr.is_null() {
                return Err(Error::Runtime(
                    "Failed to add convolution layer".to_string(),
                ));
            }
            Ok(ConvolutionLayer::from_ptr(layer_ptr))
        }
        #[cfg(feature = "mock")]
        {
            Ok(ConvolutionLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a concatenation layer
    pub fn add_concatenation(&mut self, inputs: &[&Tensor]) -> Result<ConcatenationLayer> {
        #[cfg(not(feature = "mock"))]
        {
            // Note: addConcatenation has special signature - keeping wrapper for now
            let mut input_ptrs: Vec<*mut std::ffi::c_void> =
                inputs.iter().map(|t| t.inner).collect();

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
        #[cfg(feature = "mock")]
        {
            Ok(ConcatenationLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a constant tensor
    ///
    /// # Arguments
    /// * `dims` - Dimensions of the tensor
    /// * `weights` - Raw byte data for the tensor
    /// * `data_type` - TensorRT data type (0=kFLOAT, 1=kHALF, 2=kINT8, 3=kINT32, 4=kUINT8)
    ///
    /// # Note
    /// The `weights` slice must contain the correct number of bytes for the given
    /// dimensions and data type. For example, a [2,2] float32 tensor needs 16 bytes.
    pub fn add_constant(
        &mut self,
        dims: &[i32],
        weights: &[u8],
        data_type: i32,
    ) -> Result<ConstantLayer> {
        #[cfg(not(feature = "mock"))]
        {
            // Calculate element count from dimensions
            let element_count: i64 = dims.iter().map(|&d| d as i64).product();

            // Calculate expected bytes based on data type
            let bytes_per_element = match data_type {
                0 => 4, // kFLOAT
                1 => 2, // kHALF
                2 => 1, // kINT8
                3 => 4, // kINT32
                4 => 1, // kUINT8
                _ => {
                    return Err(Error::Runtime(format!(
                        "Unsupported data type: {}",
                        data_type
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

            // Convert i32 dims to i64 and create Dims structure
            let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
            let dims_struct = trtx_sys::Dims::from_slice(&dims_i64);

            let layer_ptr = unsafe {
                trtx_sys::network_add_constant(
                    self.inner,
                    &dims_struct as *const _,
                    weights.as_ptr() as *const std::ffi::c_void,
                    data_type,
                    element_count,
                )
            };
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add constant tensor".to_string()));
            }
            Ok(ConstantLayer::from_ptr(layer_ptr))
        }
        #[cfg(feature = "mock")]
        {
            Ok(ConstantLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a SoftMax layer
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `axes` - Axes along which to apply softmax (bitwise combination)
    pub fn add_softmax(&mut self, input: &Tensor, axes: u32) -> Result<SoftMaxLayer> {
        #[cfg(not(feature = "mock"))]
        {
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

                // Set axes on the layer
                let mut layer_pin = std::pin::Pin::new_unchecked(&mut *layer_ptr);
                layer_pin.as_mut().setAxes(axes);

                layer_ptr as *mut std::ffi::c_void
            };

            Ok(SoftMaxLayer::from_ptr(layer_ptr))
        }
        #[cfg(feature = "mock")]
        {
            Ok(SoftMaxLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a Scale layer (scale, shift, power operations)
    ///
    /// Applies the formula: `output = (input * scale + shift) ^ power`
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `mode` - Scale mode:
    ///   - `0` (Uniform): Single value, applied to all elements
    ///   - `1` (Channel): Per-channel scaling (C values, where C = number of channels)
    ///   - `2` (Elementwise): Per-element scaling (same shape as input)
    /// * `shift` - Shift weights as raw bytes (float32)
    ///   - Size depends on mode: 1 value (Uniform), C values (Channel), or N values (Elementwise)
    /// * `scale` - Scale weights as raw bytes (float32)
    ///   - Size depends on mode: 1 value (Uniform), C values (Channel), or N values (Elementwise)
    /// * `power` - Power weights as raw bytes (float32)
    ///   - Size depends on mode: 1 value (Uniform), C values (Channel), or N values (Elementwise)
    ///
    /// # Note
    /// The caller must ensure the weight data has the correct size for the specified mode.
    /// Weight count is calculated automatically based on the scale mode and input dimensions.
    pub fn add_scale(
        &mut self,
        input: &Tensor,
        mode: i32,
        shift: &[u8],
        scale: &[u8],
        power: &[u8],
    ) -> Result<ScaleLayer> {
        #[cfg(not(feature = "mock"))]
        {
            // Calculate weight count based on mode and input dimensions
            let weight_count = if mode == 0 {
                // Uniform: 1 value
                1i64
            } else if mode == 1 {
                // Channel: C values (number of channels)
                let input_dims = input.dimensions()?;
                if input_dims.len() >= 4 {
                    // Format: [N, C, H, W] - channels at index 1
                    input_dims[1] as i64
                } else if input_dims.len() >= 1 {
                    // Format: [C, H, W] - channels at index 0
                    input_dims[0] as i64
                } else {
                    1
                }
            } else if mode == 2 {
                // Elementwise: total number of elements
                let input_dims = input.dimensions()?;
                input_dims.iter().map(|&d| d as i64).product()
            } else {
                return Err(Error::InvalidArgument(format!(
                    "Invalid scale mode: {}. Must be 0 (Uniform), 1 (Channel), or 2 (Elementwise)",
                    mode
                )));
            };

            let layer_ptr = unsafe {
                trtx_sys::network_add_scale(
                    self.inner,
                    input.inner,
                    mode,
                    shift.as_ptr() as *const std::ffi::c_void,
                    scale.as_ptr() as *const std::ffi::c_void,
                    power.as_ptr() as *const std::ffi::c_void,
                    weight_count,
                )
            };
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add scale layer".to_string()));
            }
            Ok(ScaleLayer::from_ptr(layer_ptr))
        }
        #[cfg(feature = "mock")]
        {
            Ok(ScaleLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a Reduce layer (reduction operations)
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `op` - Reduce operation (Sum, Prod, Max, Min, Avg)
    /// * `axes` - Axes to reduce (bitwise combination)
    /// * `keep_dims` - Whether to keep reduced dimensions
    pub fn add_reduce(
        &mut self,
        input: &Tensor,
        op: i32,
        axes: u32,
        keep_dims: bool,
    ) -> Result<ReduceLayer> {
        #[cfg(not(feature = "mock"))]
        {
            let layer_ptr = unsafe {
                let input_ref = &mut *(input.inner as *mut trtx_sys::nvinfer1::ITensor);
                let mut network_pin = crate::autocxx_helpers::cast_and_pin::<
                    trtx_sys::nvinfer1::INetworkDefinition,
                >(self.inner);

                let layer_ptr = network_pin.as_mut().addReduce(
                    std::pin::Pin::new_unchecked(input_ref),
                    std::mem::transmute::<i32, trtx_sys::nvinfer1::ReduceOperation>(op),
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
        #[cfg(feature = "mock")]
        {
            Ok(ReduceLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a Slice layer (tensor slicing)
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `start` - Start indices for each dimension
    /// * `size` - Size of slice in each dimension
    /// * `stride` - Stride for each dimension
    pub fn add_slice(
        &mut self,
        input: &Tensor,
        start: &[i32],
        size: &[i32],
        stride: &[i32],
    ) -> Result<SliceLayer> {
        #[cfg(not(feature = "mock"))]
        {
            if start.len() != size.len() || start.len() != stride.len() {
                return Err(Error::Runtime(
                    "start, size, and stride must have the same length".to_string(),
                ));
            }
            
            // Convert i32 dims to i64 and create Dims structures
            let start_i64: Vec<i64> = start.iter().map(|&d| d as i64).collect();
            let size_i64: Vec<i64> = size.iter().map(|&d| d as i64).collect();
            let stride_i64: Vec<i64> = stride.iter().map(|&d| d as i64).collect();
            
            let start_dims = trtx_sys::Dims::from_slice(&start_i64);
            let size_dims = trtx_sys::Dims::from_slice(&size_i64);
            let stride_dims = trtx_sys::Dims::from_slice(&stride_i64);
            
            let layer_ptr = unsafe {
                trtx_sys::network_add_slice(
                    self.inner,
                    input.inner,
                    &start_dims as *const _,
                    &size_dims as *const _,
                    &stride_dims as *const _,
                )
            };
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add slice layer".to_string()));
            }
            Ok(SliceLayer::from_ptr(layer_ptr))
        }
        #[cfg(feature = "mock")]
        {
            Ok(SliceLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a Resize layer (for upsampling/downsampling)
    ///
    /// Note: Use the returned layer to set resize mode and scales
    pub fn add_resize(&mut self, input: &Tensor) -> Result<ResizeLayer> {
        #[cfg(not(feature = "mock"))]
        {
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
        #[cfg(feature = "mock")]
        {
            Ok(ResizeLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a TopK layer (select top K values)
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `op` - TopK operation (0=Max, 1=Min)
    /// * `k` - Number of top elements to select
    /// * `axes` - Axes along which to apply TopK (bitwise combination)
    pub fn add_topk(&mut self, input: &Tensor, op: i32, k: i32, axes: u32) -> Result<TopKLayer> {
        #[cfg(not(feature = "mock"))]
        {
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
        #[cfg(feature = "mock")]
        {
            Ok(TopKLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a Gather layer (gather elements along an axis)
    ///
    /// # Arguments
    /// * `data` - Data tensor
    /// * `indices` - Indices tensor
    /// * `axis` - Axis along which to gather
    pub fn add_gather(
        &mut self,
        data: &Tensor,
        indices: &Tensor,
        axis: i32,
    ) -> Result<GatherLayer> {
        #[cfg(not(feature = "mock"))]
        {
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
        #[cfg(feature = "mock")]
        {
            Ok(GatherLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add a Select layer (conditional selection)
    ///
    /// Selects elements from `then_input` or `else_input` based on `condition`.
    ///
    /// # Arguments
    /// * `condition` - Boolean condition tensor
    /// * `then_input` - Tensor to select from when condition is true
    /// * `else_input` - Tensor to select from when condition is false
    pub fn add_select(
        &mut self,
        condition: &Tensor,
        then_input: &Tensor,
        else_input: &Tensor,
    ) -> Result<SelectLayer> {
        #[cfg(not(feature = "mock"))]
        {
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
        #[cfg(feature = "mock")]
        {
            Ok(SelectLayer::from_ptr(std::ptr::null_mut()))
        }
    }

    /// Add an Assertion layer
    ///
    /// Adds a runtime assertion that the condition tensor is true.
    /// If the assertion fails during execution, the engine will report an error.
    ///
    /// # Arguments
    /// * `condition` - Boolean condition tensor that must be true
    /// * `message` - Error message to display if assertion fails
    ///
    /// # Returns
    /// Returns `Ok(())` on success. The assertion layer has no output tensor.
    pub fn add_assertion(&mut self, condition: &Tensor, message: &str) -> Result<()> {
        #[cfg(not(feature = "mock"))]
        {
            let message_cstr = std::ffi::CString::new(message)?;
            let layer_ptr = unsafe {
                trtx_sys::network_add_assertion(self.inner, condition.inner, message_cstr.as_ptr())
            };
            if layer_ptr.is_null() {
                return Err(Error::Runtime("Failed to add assertion layer".to_string()));
            }
            Ok(())
        }
        #[cfg(feature = "mock")]
        {
            Ok(())
        }
    }

    /// Add a Loop construct to the network
    ///
    /// Creates an `ILoop` object for building loop constructs.
    /// The loop can be configured with trip limits, recurrence, and loop outputs.
    ///
    /// # Returns
    /// Returns a raw pointer to the `ILoop` object. The caller must manage the loop's
    /// configuration (adding loop inputs, outputs, trip limit, etc.)
    ///
    /// Note: This is a low-level API. The returned pointer is managed by the network
    /// and will be freed when the network is destroyed.
    pub fn add_loop(&mut self) -> Result<*mut std::ffi::c_void> {
        #[cfg(not(feature = "mock"))]
        {
            let loop_ptr = unsafe { trtx_sys::network_add_loop(self.inner) };
            if loop_ptr.is_null() {
                return Err(Error::Runtime("Failed to add loop".to_string()));
            }
            Ok(loop_ptr)
        }
        #[cfg(feature = "mock")]
        {
            Ok(std::ptr::null_mut())
        }
    }

    /// Add an If-conditional construct to the network
    ///
    /// Creates an `IIfConditional` object for building if-then-else constructs.
    /// The conditional can be configured with a condition and separate then/else subnetworks.
    ///
    /// # Returns
    /// Returns a raw pointer to the `IIfConditional` object. The caller must configure
    /// the conditional (setting condition, adding then/else outputs, etc.)
    ///
    /// Note: This is a low-level API. The returned pointer is managed by the network
    /// and will be freed when the network is destroyed.
    pub fn add_if_conditional(&mut self) -> Result<*mut std::ffi::c_void> {
        #[cfg(not(feature = "mock"))]
        {
            let if_ptr = unsafe { trtx_sys::network_add_if_conditional(self.inner) };
            if if_ptr.is_null() {
                return Err(Error::Runtime("Failed to add if conditional".to_string()));
            }
            Ok(if_ptr)
        }
        #[cfg(feature = "mock")]
        {
            Ok(std::ptr::null_mut())
        }
    }
}

impl Drop for NetworkDefinition {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            #[cfg(feature = "mock")]
            unsafe {
                trtx_sys::trtx_network_destroy(self.inner);
            }
            #[cfg(not(feature = "mock"))]
            unsafe {
                trtx_sys::delete_network(self.inner);
            }
        }
    }
}

unsafe impl Send for NetworkDefinition {}
