//! Mock network implementations
//!
//! Helper functions and impl blocks for mock mode.
//! Most mock implementations return null pointers or default values.

use crate::error::Result;
use crate::network::*;

//==============================================================================
// Helper functions (used by other mock modules via crate::mock::)
//==============================================================================

/// Default tensor shape for mock (e.g., [1, 3, 224, 224])
pub(crate) fn default_tensor_dimensions() -> Vec<i32> {
    vec![1, 3, 224, 224]
}

/// Default tensor name for mock
pub(crate) fn default_tensor_name() -> &'static str {
    "mock_tensor"
}

/// Default data type for mock (kFLOAT = 0)
pub(crate) fn default_tensor_type() -> i32 {
    0
}

/// Default shape for CudaEngine::get_tensor_shape (mock mode)
pub(crate) fn default_engine_tensor_shape() -> Vec<i64> {
    vec![1_i64, 1000]
}

/// Destroy network (mock mode)
pub(crate) fn destroy_network(network_ptr: *mut trtx_sys::TrtxNetworkDefinition) {
    if !network_ptr.is_null() {
        unsafe {
            trtx_sys::trtx_network_destroy(network_ptr);
        }
    }
}

//==============================================================================
// Impl blocks for network types (mock stubs)
//==============================================================================

/// Macro to implement Layer trait for mock (stub implementations)
macro_rules! impl_layer_mock {
    ($name:ident) => {
        impl Layer for $name {
            fn get_output(&self, _index: i32) -> Result<Tensor> {
                Ok(Tensor {
                    inner: std::ptr::null_mut(),
                })
            }
            fn as_ptr(&self) -> *mut std::ffi::c_void {
                self.inner
            }
        }
    };
}

impl_layer_mock!(ShuffleLayer);
impl_layer_mock!(ActivationLayer);
impl_layer_mock!(ElementWiseLayer);
impl_layer_mock!(ResizeLayer);
impl_layer_mock!(TopKLayer);
impl_layer_mock!(GatherLayer);
impl_layer_mock!(ScatterLayer);
impl_layer_mock!(SelectLayer);
impl_layer_mock!(MatrixMultiplyLayer);
impl_layer_mock!(SoftMaxLayer);
impl_layer_mock!(ReduceLayer);
impl_layer_mock!(CumulativeLayer);
impl_layer_mock!(PoolingLayer);
impl_layer_mock!(ConvolutionLayer);
impl_layer_mock!(DeconvolutionLayer);
impl_layer_mock!(QuantizeLayer);
impl_layer_mock!(DequantizeLayer);
impl_layer_mock!(ConstantLayer);
impl_layer_mock!(ConcatenationLayer);
impl_layer_mock!(ScaleLayer);
impl_layer_mock!(SliceLayer);
impl_layer_mock!(UnaryLayer);
impl_layer_mock!(IdentityLayer);
impl_layer_mock!(PaddingLayer);
impl_layer_mock!(CastLayer);

// Layer-specific impls - all no-ops for mock
impl ShuffleLayer {
    pub fn set_reshape_dimensions(&mut self, _dims: &[i32]) -> Result<()> {
        Ok(())
    }
}
impl ResizeLayer {
    pub fn set_output_dimensions(&mut self, _dims: &[i32]) -> Result<()> {
        Ok(())
    }
    pub fn set_resize_mode(&mut self, _mode: trtx_sys::ResizeMode) -> Result<()> {
        Ok(())
    }
}
impl GatherLayer {
    pub fn set_gather_mode(&mut self, _mode: trtx_sys::nvinfer1::GatherMode) -> Result<()> {
        Ok(())
    }
}
impl ScatterLayer {
    pub fn set_scatter_mode(&mut self, _mode: trtx_sys::nvinfer1::ScatterMode) -> Result<()> {
        Ok(())
    }
    pub fn set_axis(&mut self, _axis: i32) -> Result<()> {
        Ok(())
    }
}
impl ConvolutionLayer {
    pub fn set_stride(&mut self, _stride: &[i32; 2]) -> Result<()> {
        Ok(())
    }
    pub fn set_padding(&mut self, _padding: &[i32; 2]) -> Result<()> {
        Ok(())
    }
    pub fn set_dilation(&mut self, _dilation: &[i32; 2]) -> Result<()> {
        Ok(())
    }
    pub fn set_num_groups(&mut self, _num_groups: i32) -> Result<()> {
        Ok(())
    }
}
impl ConcatenationLayer {
    pub fn set_axis(&mut self, _axis: i32) -> Result<()> {
        Ok(())
    }
}

impl Tensor {
    pub fn name(&self) -> Result<String> {
        Ok(default_tensor_name().to_string())
    }
    pub fn set_name(&mut self, _name: &str) -> Result<()> {
        Ok(())
    }
    pub fn dimensions(&self) -> Result<Vec<i32>> {
        Ok(default_tensor_dimensions())
    }
    pub fn get_type(&self) -> Result<i32> {
        Ok(default_tensor_type())
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
        _name: &str,
        _data_type: trtx_sys::nvinfer1::DataType,
        _dims: &[i32],
    ) -> Result<Tensor> {
        Ok(Tensor {
            inner: std::ptr::null_mut(),
        })
    }
    pub fn mark_output(&mut self, _tensor: &Tensor) -> Result<()> {
        Ok(())
    }
    pub fn get_nb_inputs(&self) -> i32 {
        0
    }
    pub fn get_nb_outputs(&self) -> i32 {
        0
    }
    pub fn get_input(&self, _index: i32) -> Result<Tensor> {
        Ok(Tensor {
            inner: std::ptr::null_mut(),
        })
    }
    pub fn get_output(&self, _index: i32) -> Result<Tensor> {
        Ok(Tensor {
            inner: std::ptr::null_mut(),
        })
    }

    pub fn add_activation(
        &mut self,
        _input: &Tensor,
        _activation_type: trtx_sys::nvinfer1::ActivationType,
    ) -> Result<ActivationLayer> {
        Ok(ActivationLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_unary(
        &mut self,
        _input: &Tensor,
        _op: trtx_sys::nvinfer1::UnaryOperation,
    ) -> Result<UnaryLayer> {
        Ok(UnaryLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_identity(&mut self, _input: &Tensor) -> Result<IdentityLayer> {
        Ok(IdentityLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_cast(
        &mut self,
        _input: &Tensor,
        _to_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<CastLayer> {
        Ok(CastLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_elementwise(
        &mut self,
        _input1: &Tensor,
        _input2: &Tensor,
        _op: trtx_sys::nvinfer1::ElementWiseOperation,
    ) -> Result<ElementWiseLayer> {
        Ok(ElementWiseLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_pooling(
        &mut self,
        _input: &Tensor,
        _pooling_type: trtx_sys::nvinfer1::PoolingType,
        _window_size: &[i32; 2],
    ) -> Result<PoolingLayer> {
        Ok(PoolingLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_shuffle(&mut self, _input: &Tensor) -> Result<ShuffleLayer> {
        Ok(ShuffleLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_matrix_multiply(
        &mut self,
        _input0: &Tensor,
        _op0: i32,
        _input1: &Tensor,
        _op1: i32,
    ) -> Result<MatrixMultiplyLayer> {
        Ok(MatrixMultiplyLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_convolution(
        &mut self,
        _input: &Tensor,
        _nb_output_maps: i32,
        _kernel_size: &[i32; 2],
        _kernel_weights: &[u8],
        _bias_weights: Option<&[u8]>,
    ) -> Result<ConvolutionLayer> {
        Ok(ConvolutionLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_deconvolution(
        &mut self,
        _input: &Tensor,
        _nb_output_maps: i32,
        _kernel_size: &[i32; 2],
        _kernel_weights: &[u8],
        _bias_weights: Option<&[u8]>,
    ) -> Result<DeconvolutionLayer> {
        Ok(DeconvolutionLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_concatenation(&mut self, _inputs: &[&Tensor]) -> Result<ConcatenationLayer> {
        Ok(ConcatenationLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_constant(
        &mut self,
        _dims: &[i32],
        _weights: &[u8],
        _data_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<ConstantLayer> {
        Ok(ConstantLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_softmax(&mut self, _input: &Tensor, _axes: u32) -> Result<SoftMaxLayer> {
        Ok(SoftMaxLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_scale(
        &mut self,
        _input: &Tensor,
        _mode: i32,
        _shift: &[u8],
        _scale: &[u8],
        _power: &[u8],
    ) -> Result<ScaleLayer> {
        Ok(ScaleLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_reduce(
        &mut self,
        _input: &Tensor,
        _op: trtx_sys::nvinfer1::ReduceOperation,
        _axes: u32,
        _keep_dims: bool,
    ) -> Result<ReduceLayer> {
        Ok(ReduceLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_cumulative(
        &mut self,
        _input: &Tensor,
        _axis: i32,
        _op: trtx_sys::nvinfer1::CumulativeOperation,
        _exclusive: bool,
        _reverse: bool,
    ) -> Result<CumulativeLayer> {
        Ok(CumulativeLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_cumulative_with_axis_tensor(
        &mut self,
        _input: &Tensor,
        _axis_tensor: &Tensor,
        _op: trtx_sys::nvinfer1::CumulativeOperation,
        _exclusive: bool,
        _reverse: bool,
    ) -> Result<CumulativeLayer> {
        Ok(CumulativeLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_slice(
        &mut self,
        _input: &Tensor,
        _start: &[i32],
        _size: &[i32],
        _stride: &[i32],
    ) -> Result<SliceLayer> {
        Ok(SliceLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_resize(&mut self, _input: &Tensor) -> Result<ResizeLayer> {
        Ok(ResizeLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_topk(
        &mut self,
        _input: &Tensor,
        _op: i32,
        _k: i32,
        _axes: u32,
    ) -> Result<TopKLayer> {
        Ok(TopKLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_gather(
        &mut self,
        _data: &Tensor,
        _indices: &Tensor,
        _axis: i32,
    ) -> Result<GatherLayer> {
        Ok(GatherLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_scatter(
        &mut self,
        _data: &Tensor,
        _indices: &Tensor,
        _updates: &Tensor,
        _mode: trtx_sys::nvinfer1::ScatterMode,
    ) -> Result<ScatterLayer> {
        Ok(ScatterLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_quantize(
        &mut self,
        _input: &Tensor,
        _scale: &Tensor,
        _output_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<QuantizeLayer> {
        Ok(QuantizeLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_dequantize(
        &mut self,
        _input: &Tensor,
        _scale: &Tensor,
        _output_type: trtx_sys::nvinfer1::DataType,
    ) -> Result<DequantizeLayer> {
        Ok(DequantizeLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_select(
        &mut self,
        _condition: &Tensor,
        _then_input: &Tensor,
        _else_input: &Tensor,
    ) -> Result<SelectLayer> {
        Ok(SelectLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_padding(
        &mut self,
        _input: &Tensor,
        _pre_padding: &[i32],
        _post_padding: &[i32],
    ) -> Result<PaddingLayer> {
        Ok(PaddingLayer::from_ptr(std::ptr::null_mut()))
    }
    pub fn add_assertion(&mut self, _condition: &Tensor, _message: &str) -> Result<()> {
        Ok(())
    }
    pub fn add_loop(&mut self) -> Result<*mut std::ffi::c_void> {
        Ok(std::ptr::null_mut())
    }
    pub fn add_if_conditional(&mut self) -> Result<*mut std::ffi::c_void> {
        Ok(std::ptr::null_mut())
    }
}

impl Drop for NetworkDefinition {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            destroy_network(self.inner as *mut trtx_sys::TrtxNetworkDefinition);
        }
    }
}

unsafe impl Send for NetworkDefinition {}
