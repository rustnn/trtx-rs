//! Builder for creating TensorRT engines

use crate::error::{Error, Result};
use crate::logger::Logger;
use trtx_sys::*;

/// Network definition builder flags
pub mod network_flags {
    /// Explicit batch sizes
    pub const EXPLICIT_BATCH: u32 = 1 << 0;
}

/// Memory pool types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MemoryPoolType {
    /// Workspace memory
    Workspace = 0,
    /// DLA managed SRAM
    DlaManagedSram = 1,
    /// DLA local DRAM
    DlaLocalDram = 2,
    /// DLA global DRAM
    DlaGlobalDram = 3,
}

/// Network definition for building TensorRT engines
pub struct NetworkDefinition {
    inner: *mut TrtxNetworkDefinition,
}

impl NetworkDefinition {
    /// Get the raw pointer (for internal use)
    pub(crate) fn as_ptr(&self) -> *mut TrtxNetworkDefinition {
        self.inner
    }
}

impl Drop for NetworkDefinition {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_network_destroy(self.inner);
            }
        }
    }
}

unsafe impl Send for NetworkDefinition {}

/// Builder configuration
pub struct BuilderConfig {
    inner: *mut TrtxBuilderConfig,
}

impl BuilderConfig {
    /// Set memory pool limit
    pub fn set_memory_pool_limit(&mut self, pool: MemoryPoolType, size: usize) -> Result<()> {
        let mut error_msg = [0i8; 1024];

        let result = unsafe {
            trtx_builder_config_set_memory_pool_limit(
                self.inner,
                pool as i32,
                size,
                error_msg.as_mut_ptr(),
                error_msg.len(),
            )
        };

        if result != TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        Ok(())
    }

    /// Get the raw pointer (for internal use)
    pub(crate) fn as_ptr(&self) -> *mut TrtxBuilderConfig {
        self.inner
    }
}

impl Drop for BuilderConfig {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_builder_config_destroy(self.inner);
            }
        }
    }
}

unsafe impl Send for BuilderConfig {}

/// Builder for creating optimized TensorRT engines
pub struct Builder<'a> {
    inner: *mut TrtxBuilder,
    _logger: &'a Logger,
}

impl<'a> Builder<'a> {
    /// Create a new builder
    pub fn new(logger: &'a Logger) -> Result<Self> {
        let mut builder_ptr: *mut TrtxBuilder = std::ptr::null_mut();
        let mut error_msg = [0i8; 1024];

        let result = unsafe {
            trtx_builder_create(
                logger.as_ptr(),
                &mut builder_ptr,
                error_msg.as_mut_ptr(),
                error_msg.len(),
            )
        };

        if result != TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        Ok(Builder {
            inner: builder_ptr,
            _logger: logger,
        })
    }

    /// Create a network definition
    pub fn create_network(&self, flags: u32) -> Result<NetworkDefinition> {
        let mut network_ptr: *mut TrtxNetworkDefinition = std::ptr::null_mut();
        let mut error_msg = [0i8; 1024];

        let result = unsafe {
            trtx_builder_create_network(
                self.inner,
                flags,
                &mut network_ptr,
                error_msg.as_mut_ptr(),
                error_msg.len(),
            )
        };

        if result != TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        Ok(NetworkDefinition { inner: network_ptr })
    }

    /// Create a builder configuration
    pub fn create_config(&self) -> Result<BuilderConfig> {
        let mut config_ptr: *mut TrtxBuilderConfig = std::ptr::null_mut();
        let mut error_msg = [0i8; 1024];

        let result = unsafe {
            trtx_builder_create_builder_config(
                self.inner,
                &mut config_ptr,
                error_msg.as_mut_ptr(),
                error_msg.len(),
            )
        };

        if result != TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        Ok(BuilderConfig { inner: config_ptr })
    }

    /// Build a serialized network (engine)
    pub fn build_serialized_network(
        &self,
        network: &NetworkDefinition,
        config: &BuilderConfig,
    ) -> Result<Vec<u8>> {
        let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut size: usize = 0;
        let mut error_msg = [0i8; 1024];

        let result = unsafe {
            trtx_builder_build_serialized_network(
                self.inner,
                network.as_ptr(),
                config.as_ptr(),
                &mut data_ptr,
                &mut size,
                error_msg.as_mut_ptr(),
                error_msg.len(),
            )
        };

        if result != TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        // Copy data to Vec and free C buffer
        let data = unsafe {
            let slice = std::slice::from_raw_parts(data_ptr as *const u8, size);
            let vec = slice.to_vec();
            trtx_free_buffer(data_ptr);
            vec
        };

        Ok(data)
    }
}

impl Drop for Builder<'_> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_builder_destroy(self.inner);
            }
        }
    }
}

unsafe impl Send for Builder<'_> {}
