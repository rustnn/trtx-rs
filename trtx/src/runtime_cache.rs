//! Runtime cache for TensorRT JIT compilation (serialize / deserialize).
//!
//! [`RuntimeCache`] wraps [`trtx_sys::nvinfer1::IRuntimeCache`] (C++ [`nvinfer1::IRuntimeCache`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_runtime_cache.html).

use std::marker::PhantomData;

use crate::error::{PropertySetAttempt, Result};
use crate::host_memory::HostMemory;
use crate::Error;
use cxx::UniquePtr;
use trtx_sys::nvinfer1::{self, IRuntimeCache};

/// [`trtx_sys::nvinfer1::IRuntimeCache`] — C++ [`nvinfer1::IRuntimeCache`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_runtime_cache.html).
pub struct RuntimeCache<'engine> {
    pub(crate) inner: UniquePtr<IRuntimeCache>,
    _engine: PhantomData<&'engine nvinfer1::ICudaEngine>,
}

impl std::fmt::Debug for RuntimeCache<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeCache")
            .field("inner", &format!("{:x}", self.inner.as_ptr() as usize))
            .finish_non_exhaustive()
    }
}

impl<'engine> RuntimeCache<'engine> {
    pub(crate) fn new(cache: *mut nvinfer1::IRuntimeCache) -> Result<Self> {
        #[cfg(not(feature = "mock"))]
        if cache.is_null() {
            return Err(Error::RuntimeCacheCreationFailed);
        }
        Ok(Self {
            inner: unsafe { UniquePtr::from_raw(cache) },
            _engine: Default::default(),
        })
    }

    /// See [IRuntimeCache::serialize].
    pub fn serialize(&self) -> Result<HostMemory<'engine>> {
        #[cfg(not(feature = "mock"))]
        {
            let host_mem = unsafe { self.inner.serialize().as_mut() }
                .ok_or_else(|| Error::Runtime("Failed to serialize IRuntimeCache".to_string()))?;
            Ok(unsafe { HostMemory::from_raw(host_mem) })
        }
        #[cfg(feature = "mock")]
        Ok(unsafe { HostMemory::from_raw(std::ptr::null_mut()) })
    }

    /// See [IRuntimeCache::deserialize].
    pub fn deserialize(&mut self, blob: &[u8]) -> Result<()> {
        if cfg!(not(feature = "mock")) {
            if unsafe {
                self.inner
                    .pin_mut()
                    .deserialize(blob.as_ptr() as *const autocxx::c_void, blob.len())
            } {
                Ok(())
            } else {
                Err(Error::FailedToSetProperty(
                    PropertySetAttempt::RuntimeCacheDeserialize,
                ))
            }
        } else {
            Ok(())
        }
    }

    /// See [IRuntimeCache::reset].
    pub fn reset(&mut self) -> Result<()> {
        if cfg!(not(feature = "mock")) {
            if self.inner.pin_mut().reset() {
                Ok(())
            } else {
                Err(Error::FailedToResetRuntimeCache)
            }
        } else {
            Ok(())
        }
    }
}
