//! Network definition for building TensorRT engines

use crate::error::{Error, Result};

/// Tensor handle (opaque pointer)
pub struct Tensor {
    inner: *mut std::ffi::c_void,
}

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
            // Keep wrapper - creating Dims object is complex
            let name_cstr = std::ffi::CString::new(name)?;
            let tensor_ptr = unsafe {
                trtx_sys::network_add_input(
                    self.inner,
                    name_cstr.as_ptr(),
                    data_type,
                    dims.as_ptr(),
                    dims.len() as i32,
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
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(self.inner)
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
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(self.inner)
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
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(self.inner)
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
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(self.inner)
                    .getInput(index)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime(format!(
                    "Failed to get input at index {}",
                    index
                )));
            }
            Ok(Tensor { inner: tensor_ptr as *mut _ })
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
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::INetworkDefinition>(self.inner)
                    .getOutput(index)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime(format!(
                    "Failed to get output at index {}",
                    index
                )));
            }
            Ok(Tensor { inner: tensor_ptr as *mut _ })
        }
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Add an activation layer
    pub fn add_activation(&mut self, input: &Tensor, activation_type: i32) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_activation(self.inner, input.inner, activation_type)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add activation layer".to_string()));
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

    /// Add an elementwise operation layer
    pub fn add_elementwise(&mut self, input1: &Tensor, input2: &Tensor, op: i32) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_elementwise(self.inner, input1.inner, input2.inner, op)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime(
                    "Failed to add elementwise layer".to_string(),
                ));
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

    /// Add a pooling layer
    pub fn add_pooling(
        &mut self,
        input: &Tensor,
        pooling_type: i32,
        window_size: &[i32; 2],
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_pooling(
                    self.inner,
                    input.inner,
                    pooling_type,
                    window_size.as_ptr(),
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add pooling layer".to_string()));
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

    /// Add a shuffle layer (for reshaping/transposing)
    pub fn add_shuffle(&mut self, input: &Tensor) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            // Keep wrapper - layer inheritance makes autocxx usage complex
            let tensor_ptr = unsafe { trtx_sys::network_add_shuffle(self.inner, input.inner) };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add shuffle layer".to_string()));
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

    /// Add a matrix multiply layer
    pub fn add_matrix_multiply(
        &mut self,
        input0: &Tensor,
        op0: i32,
        input1: &Tensor,
        op1: i32,
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_matrix_multiply(
                    self.inner,
                    input0.inner,
                    op0,
                    input1.inner,
                    op1,
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime(
                    "Failed to add matrix multiply layer".to_string(),
                ));
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

    /// Add a convolution layer
    pub fn add_convolution(
        &mut self,
        input: &Tensor,
        nb_output_maps: i32,
        kernel_size: &[i32; 2],
        kernel_weights: &[u8],
        bias_weights: Option<&[u8]>,
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let bias_ptr = bias_weights
                .map(|b| b.as_ptr() as *const std::ffi::c_void)
                .unwrap_or(std::ptr::null());

            let tensor_ptr = unsafe {
                trtx_sys::network_add_convolution(
                    self.inner,
                    input.inner,
                    nb_output_maps,
                    kernel_size.as_ptr(),
                    kernel_weights.as_ptr() as *const std::ffi::c_void,
                    bias_ptr,
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime(
                    "Failed to add convolution layer".to_string(),
                ));
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

    /// Add a concatenation layer
    pub fn add_concatenation(&mut self, inputs: &[&Tensor]) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let mut input_ptrs: Vec<*mut std::ffi::c_void> =
                inputs.iter().map(|t| t.inner).collect();

            let tensor_ptr = unsafe {
                trtx_sys::network_add_concatenation(
                    self.inner,
                    input_ptrs.as_mut_ptr(),
                    inputs.len() as i32,
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime(
                    "Failed to add concatenation layer".to_string(),
                ));
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
    pub fn add_constant(&mut self, dims: &[i32], weights: &[u8], data_type: i32) -> Result<Tensor> {
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
                _ => return Err(Error::Runtime(format!("Unsupported data type: {}", data_type))),
            };
            
            let expected_bytes = element_count * bytes_per_element;
            if weights.len() as i64 != expected_bytes {
                return Err(Error::Runtime(format!(
                    "Weight size mismatch: expected {} bytes, got {} bytes",
                    expected_bytes,
                    weights.len()
                )));
            }
            
            let tensor_ptr = unsafe {
                trtx_sys::network_add_constant(
                    self.inner,
                    dims.as_ptr(),
                    dims.len() as i32,
                    weights.as_ptr() as *const std::ffi::c_void,
                    data_type,
                    element_count,
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add constant tensor".to_string()));
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

    /// Add a SoftMax layer
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `axes` - Axes along which to apply softmax (bitwise combination)
    pub fn add_softmax(&mut self, input: &Tensor, axes: u32) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr =
                unsafe { trtx_sys::network_add_softmax(self.inner, input.inner, axes) };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add softmax layer".to_string()));
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

    /// Add a Scale layer (scale, shift, power operations)
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `mode` - Scale mode (0=Uniform, 1=Channel, 2=Elementwise)
    /// * `shift` - Shift weights
    /// * `scale` - Scale weights
    /// * `power` - Power weights
    pub fn add_scale(
        &mut self,
        input: &Tensor,
        mode: i32,
        shift: &[u8],
        scale: &[u8],
        power: &[u8],
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_scale(
                    self.inner,
                    input.inner,
                    mode,
                    shift.as_ptr() as *const std::ffi::c_void,
                    scale.as_ptr() as *const std::ffi::c_void,
                    power.as_ptr() as *const std::ffi::c_void,
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add scale layer".to_string()));
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
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_reduce(self.inner, input.inner, op, axes, keep_dims)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add reduce layer".to_string()));
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
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            if start.len() != size.len() || start.len() != stride.len() {
                return Err(Error::Runtime(
                    "start, size, and stride must have the same length".to_string(),
                ));
            }
            let tensor_ptr = unsafe {
                trtx_sys::network_add_slice(
                    self.inner,
                    input.inner,
                    start.as_ptr(),
                    size.as_ptr(),
                    stride.as_ptr(),
                    start.len() as i32,
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add slice layer".to_string()));
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

    /// Add a Resize layer (for upsampling/downsampling)
    ///
    /// Note: Use the returned layer to set resize mode and scales
    pub fn add_resize(&mut self, input: &Tensor) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe { trtx_sys::network_add_resize(self.inner, input.inner) };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add resize layer".to_string()));
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

    /// Add a TopK layer (select top K values)
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `op` - TopK operation (0=Max, 1=Min)
    /// * `k` - Number of top elements to select
    /// * `axes` - Axes along which to apply TopK (bitwise combination)
    pub fn add_topk(&mut self, input: &Tensor, op: i32, k: i32, axes: u32) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr =
                unsafe { trtx_sys::network_add_topk(self.inner, input.inner, op, k, axes) };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add topk layer".to_string()));
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

    /// Add a Gather layer (gather elements along an axis)
    ///
    /// # Arguments
    /// * `data` - Data tensor
    /// * `indices` - Indices tensor
    /// * `axis` - Axis along which to gather
    pub fn add_gather(&mut self, data: &Tensor, indices: &Tensor, axis: i32) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_gather(self.inner, data.inner, indices.inner, axis)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add gather layer".to_string()));
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
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_select(
                    self.inner,
                    condition.inner,
                    then_input.inner,
                    else_input.inner,
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add select layer".to_string()));
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
