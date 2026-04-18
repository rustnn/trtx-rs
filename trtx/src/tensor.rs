use std::pin::Pin;

use crate::{network::check_network, Error, NetworkDefinition, Result};
use trtx_sys::{nvinfer1, DataType};

/// [`trtx_sys::nvinfer1::ITensor`] — C++ [`nvinfer1::ITensor`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_tensor.html).
#[derive(Clone, Copy)]
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
    pub(crate) fn pin_mut(&self) -> Pin<&mut nvinfer1::ITensor> {
        unsafe { Pin::new_unchecked(self.inner.as_mut().unwrap()) }
    }
    pub(crate) fn as_ref(&self) -> &nvinfer1::ITensor {
        unsafe { self.inner.as_ref().unwrap() }
    }
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn as_mut(&self) -> &mut nvinfer1::ITensor {
        unsafe { self.inner.as_mut().unwrap() }
    }

    /// See [nvinfer1::ITensor::getName]
    pub fn name(&self, network: &NetworkDefinition) -> Result<String> {
        check_network!(network, self);
        let name_ptr = self.as_ref().getName();
        if name_ptr.is_null() {
            return Err(Error::Runtime("Failed to get tensor name".to_string()));
        }
        unsafe { Ok(std::ffi::CStr::from_ptr(name_ptr).to_str()?.to_string()) }
    }

    /// See [nvinfer1::ITensor::setName]
    pub fn set_name(&self, network: &'_ mut NetworkDefinition, name: &str) -> Result<()> {
        check_network!(network, self);
        let name_cstr = std::ffi::CString::new(name)?;
        unsafe {
            self.pin_mut().setName(name_cstr.as_ptr());
        }
        Ok(())
    }

    /// See [nvinfer1::ITensor::setDimensionName]
    pub fn set_dimension_name(
        &self,
        network: &'_ mut NetworkDefinition,
        index: i32,
        name: &str,
    ) -> Result<()> {
        check_network!(network, self);
        let name_cstr = std::ffi::CString::new(name)?;
        unsafe {
            self.pin_mut().setDimensionName(index, name_cstr.as_ptr());
        }
        Ok(())
    }

    /// See [nvinfer1::ITensor::getDimensions]
    pub fn dimensions(&self, network: &NetworkDefinition) -> Result<Vec<i64>> {
        check_network!(network, self);
        let result = self.as_ref().getDimensions();
        Ok(result.d[..result.nbDims as usize].to_vec())
    }

    /// See [nvinfer1::ITensor::isExecutionTensor]
    pub fn is_execution_tensor(&self, network: &NetworkDefinition) -> bool {
        check_network!(network, self);
        self.as_ref().isExecutionTensor()
    }

    /// See [nvinfer1::ITensor::isShapeTensor]
    pub fn is_shape_tensor(&self, network: &NetworkDefinition) -> bool {
        check_network!(network, self);
        self.as_ref().isShapeTensor()
    }

    /// See [nvinfer1::ITensor::isNetworkInput]
    pub fn is_network_input(&self, network: &NetworkDefinition) -> bool {
        check_network!(network, self);
        self.as_ref().isNetworkInput()
    }

    /// See [nvinfer1::ITensor::isNetworkOutput]
    pub fn is_network_output(&self, network: &NetworkDefinition) -> bool {
        check_network!(network, self);
        self.as_ref().isNetworkOutput()
    }

    /// See [nvinfer1::ITensor::getType]
    pub fn get_type(&self, network: &NetworkDefinition) -> DataType {
        check_network!(network, self);
        self.as_ref().getType().into()
    }

    /// Set allowed tensor formats (bitmask of TensorFormat). E.g. 1u32 << TensorFormat::kHWC for channels-last.
    /// TensorRT may insert reformat layers when connecting tensors with different formats.
    ///
    /// See [nvinfer1::ITensor::setAllowedFormats]
    pub fn set_allowed_formats(
        &mut self,
        network: &mut NetworkDefinition,
        formats: u32,
    ) -> Result<()> {
        check_network!(network, self);
        self.pin_mut().setAllowedFormats(formats);
        Ok(())
    }
}
