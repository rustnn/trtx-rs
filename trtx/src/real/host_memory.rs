use core::slice;
use std::ops::Deref;
use std::pin::Pin;
use std::ptr;

use super::builder::Builder;
use trtx_sys::nvinfer1::IHostMemory;
use trtx_sys::DataType;

pub struct HostMemory<'builder> {
    pub(crate) inner: Pin<&'builder mut IHostMemory>,
}

impl<'builder> HostMemory<'builder> {
    /// assumes ownership of ref
    pub(crate) unsafe fn from_raw_ref(
        _builder: &'builder Builder,
        ptr: &'builder mut IHostMemory,
    ) -> Self {
        unsafe {
            HostMemory {
                inner: Pin::new_unchecked(ptr),
            }
        }
    }

    pub fn data_type(&self) -> DataType {
        self.inner.as_ref().type_().into()
    }
}

impl<'memory> AsRef<[u8]> for HostMemory<'memory> {
    fn as_ref(&self) -> &'memory [u8] {
        unsafe {
            slice::from_raw_parts(
                self.inner.as_ref().data() as *const u8,
                self.inner.as_ref().size(),
            )
        }
    }
}

impl<'builder> Deref for HostMemory<'builder> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // You can leverage your existing AsRef implementation here
        self.as_ref()
    }
}

impl Drop for HostMemory<'_> {
    fn drop(&mut self) {
        unsafe { ptr::drop_in_place(self.inner.as_mut().get_unchecked_mut()) };
    }
}
