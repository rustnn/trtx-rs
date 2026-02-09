//! Mock builder implementations

use crate::builder::MemoryPoolType;
use crate::error::Result;
use crate::logger::Logger;
use crate::network::NetworkDefinition;

/// Builder configuration (mock mode)
pub struct BuilderConfig {
    pub(crate) inner: *mut trtx_sys::TrtxBuilderConfig,
}

impl BuilderConfig {
    pub fn set_memory_pool_limit(&mut self, pool: MemoryPoolType, size: usize) -> Result<()> {
        set_memory_pool_limit(self.inner, pool as i32, size)
    }

    pub(crate) fn as_ptr(&self) -> *mut trtx_sys::TrtxBuilderConfig {
        self.inner
    }
}

impl Drop for BuilderConfig {
    fn drop(&mut self) {
        destroy_config(self.inner);
    }
}

unsafe impl Send for BuilderConfig {}

/// Builder (mock mode)
pub struct Builder<'a> {
    inner: *mut trtx_sys::TrtxBuilder,
    _logger: &'a Logger,
}

impl<'a> Builder<'a> {
    pub fn new(logger: &'a Logger) -> Result<Self> {
        let builder_ptr = trtx_builder_create(logger.as_ptr())?;
        Ok(Builder {
            inner: builder_ptr,
            _logger: logger,
        })
    }

    pub fn create_network(&self, flags: u32) -> Result<NetworkDefinition> {
        create_network(self.inner, flags)
    }

    pub fn create_config(&self) -> Result<BuilderConfig> {
        let config_ptr = create_config(self.inner)?;
        Ok(BuilderConfig { inner: config_ptr })
    }

    pub fn build_serialized_network(
        &self,
        network: &mut NetworkDefinition,
        config: &mut BuilderConfig,
    ) -> Result<Vec<u8>> {
        build_serialized_network(self.inner, network.as_mut_ptr(), config.as_ptr())
    }
}

impl Drop for Builder<'_> {
    fn drop(&mut self) {
        destroy_builder(self.inner);
    }
}

unsafe impl Send for Builder<'_> {}

//------------------------------------------------------------------------------
// Helper functions (used by above impls)
//------------------------------------------------------------------------------

/// Create builder via FFI (mock mode)
pub(crate) fn trtx_builder_create(logger_ptr: *mut trtx_sys::TrtxLogger) -> Result<*mut trtx_sys::TrtxBuilder> {
    let mut builder_ptr: *mut trtx_sys::TrtxBuilder = std::ptr::null_mut();
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_builder_create(
            logger_ptr,
            &mut builder_ptr,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(super::from_ffi(result, &error_msg));
    }

    Ok(builder_ptr)
}

fn set_memory_pool_limit(
    config_ptr: *mut trtx_sys::TrtxBuilderConfig,
    pool: i32,
    size: usize,
) -> Result<()> {
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_builder_config_set_memory_pool_limit(
            config_ptr,
            pool,
            size,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(super::from_ffi(result, &error_msg));
    }

    Ok(())
}

fn destroy_config(config_ptr: *mut trtx_sys::TrtxBuilderConfig) {
    if !config_ptr.is_null() {
        unsafe {
            trtx_sys::trtx_builder_config_destroy(config_ptr);
        }
    }
}

fn create_network(
    builder_ptr: *mut trtx_sys::TrtxBuilder,
    flags: u32,
) -> Result<NetworkDefinition> {
    let mut network_ptr: *mut trtx_sys::TrtxNetworkDefinition = std::ptr::null_mut();
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_builder_create_network(
            builder_ptr,
            flags,
            &mut network_ptr,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(super::from_ffi(result, &error_msg));
    }

    Ok(NetworkDefinition::from_ptr(network_ptr as *mut std::ffi::c_void))
}

fn create_config(
    builder_ptr: *mut trtx_sys::TrtxBuilder,
) -> Result<*mut trtx_sys::TrtxBuilderConfig> {
    let mut config_ptr: *mut trtx_sys::TrtxBuilderConfig = std::ptr::null_mut();
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_builder_create_builder_config(
            builder_ptr,
            &mut config_ptr,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(super::from_ffi(result, &error_msg));
    }

    Ok(config_ptr)
}

fn build_serialized_network(
    builder_ptr: *mut trtx_sys::TrtxBuilder,
    network_ptr: *mut std::ffi::c_void,
    config_ptr: *mut trtx_sys::TrtxBuilderConfig,
) -> Result<Vec<u8>> {
    let network_ptr = network_ptr as *mut trtx_sys::TrtxNetworkDefinition;
    let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut size: usize = 0;
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_builder_build_serialized_network(
            builder_ptr,
            network_ptr,
            config_ptr,
            &mut data_ptr,
            &mut size,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(super::from_ffi(result, &error_msg));
    }

    // Copy data to Vec and free C buffer
    let data = unsafe {
        let slice = std::slice::from_raw_parts(data_ptr as *const u8, size);
        let vec = slice.to_vec();
        trtx_sys::trtx_free_buffer(data_ptr);
        vec
    };

    Ok(data)
}

fn destroy_builder(builder_ptr: *mut trtx_sys::TrtxBuilder) {
    if !builder_ptr.is_null() {
        unsafe {
            trtx_sys::trtx_builder_destroy(builder_ptr);
        }
    }
}
