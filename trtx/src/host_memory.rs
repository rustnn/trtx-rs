//! Host memory buffer returned by the TensorRT builder (serialized engines, etc.).
//!
//! [`HostMemory`] wraps [`trtx_sys::nvinfer1::IHostMemory`] (C++ [`nvinfer1::IHostMemory`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_host_memory.html)).

use core::slice;
use cxx::UniquePtr;
use std::marker::PhantomData;
use std::ops::Deref;
use trtx_sys::nvinfer1::{self};
use trtx_sys::DataType;

/// [`trtx_sys::nvinfer1::IHostMemory`] — C++ [`nvinfer1::IHostMemory`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_host_memory.html).
pub struct HostMemory<'builder> {
    pub(crate) inner: UniquePtr<nvinfer1::IHostMemory>,
    _builder: PhantomData<&'builder nvinfer1::IBuilder>,
    #[cfg(feature = "mock")]
    mock_data: Vec<u8>,
}

impl<'builder> HostMemory<'builder> {
    /// assumes ownership of ref
    pub(crate) unsafe fn from_raw(ptr: *mut nvinfer1::IHostMemory) -> Self {
        unsafe {
            HostMemory {
                inner: UniquePtr::from_raw(ptr),
                _builder: Default::default(),
                #[cfg(feature = "mock")]
                mock_data: Default::default(),
            }
        }
    }

    pub fn data_type(&self) -> DataType {
        if cfg!(feature = "mock") {
            DataType::kINT8
        } else {
            self.inner.type_().into()
        }
    }
}

#[cfg(feature = "mock")]
impl<'memory> AsRef<[u8]> for HostMemory<'memory> {
    fn as_ref(&self) -> &[u8] {
        &self.mock_data
    }
}
#[cfg(not(feature = "mock"))]
impl<'memory> AsRef<[u8]> for HostMemory<'memory> {
    fn as_ref(&self) -> &'memory [u8] {
        unsafe { slice::from_raw_parts(self.inner.data() as *const u8, self.inner.size()) }
    }
}

impl<'builder> Deref for HostMemory<'builder> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}
