use core::slice;
use cxx::UniquePtr;
use std::marker::PhantomData;
use std::ops::Deref;
use trtx_sys::nvinfer1::{self};
use trtx_sys::DataType;

pub struct HostMemory<'builder> {
    pub(crate) inner: UniquePtr<nvinfer1::IHostMemory>,
    _builder: PhantomData<&'builder nvinfer1::IBuilder>,
}

impl<'builder> HostMemory<'builder> {
    /// assumes ownership of ref
    pub(crate) unsafe fn from_raw_ref(ptr: *mut nvinfer1::IHostMemory) -> Self {
        unsafe {
            HostMemory {
                inner: UniquePtr::from_raw(ptr),
                _builder: Default::default(),
            }
        }
    }

    pub fn data_type(&self) -> DataType {
        self.inner.type_().into()
    }
}

impl<'memory> AsRef<[u8]> for HostMemory<'memory> {
    fn as_ref(&self) -> &'memory [u8] {
        unsafe { slice::from_raw_parts(self.inner.data() as *const u8, self.inner.size()) }
    }
}

impl<'builder> Deref for HostMemory<'builder> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // You can leverage your existing AsRef implementation here
        self.as_ref()
    }
}
