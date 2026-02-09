//! Real TensorRT builder implementation

use crate::builder::MemoryPoolType;
use crate::error::{Error, Result};
use crate::logger::Logger;
use crate::network::NetworkDefinition;

/// Builder configuration (real mode)
pub struct BuilderConfig {
    inner: *mut std::ffi::c_void,
}

impl BuilderConfig {
    pub fn set_memory_pool_limit(&mut self, pool: MemoryPoolType, size: usize) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid builder config".to_string()));
        }
        let trt_pool = match pool {
            MemoryPoolType::Workspace => 0,
            MemoryPoolType::DlaManagedSram => 1,
            MemoryPoolType::DlaLocalDram => 2,
            MemoryPoolType::DlaGlobalDram => 3,
        };
        unsafe {
            trtx_sys::builder_config_set_memory_pool_limit(self.inner, trt_pool, size);
        }
        Ok(())
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.inner
    }
}

impl Drop for BuilderConfig {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_sys::delete_config(self.inner);
            }
        }
    }
}

unsafe impl Send for BuilderConfig {}

/// Builder (real mode)
pub struct Builder<'a> {
    inner: *mut std::ffi::c_void,
    _logger: &'a Logger,
}

impl<'a> Builder<'a> {
    pub fn new(logger: &'a Logger) -> Result<Self> {
        let logger_ptr = logger.as_logger_ptr();
        let builder_ptr = unsafe { trtx_sys::create_infer_builder(logger_ptr) };
        if builder_ptr.is_null() {
            return Err(Error::Runtime("Failed to create builder".to_string()));
        }
        Ok(Builder {
            inner: builder_ptr,
            _logger: logger,
        })
    }

    pub fn create_network(&self, flags: u32) -> Result<NetworkDefinition> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid builder".to_string()));
        }
        let network_ptr = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::IBuilder>(self.inner)
                .createNetworkV2(flags)
        };
        if network_ptr.is_null() {
            return Err(Error::Runtime("Failed to create network".to_string()));
        }
        Ok(NetworkDefinition::from_ptr(network_ptr as *mut _))
    }

    pub fn create_config(&self) -> Result<BuilderConfig> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid builder".to_string()));
        }
        let config_ptr = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::IBuilder>(self.inner)
                .createBuilderConfig()
        };
        if config_ptr.is_null() {
            return Err(Error::Runtime(
                "Failed to create builder config".to_string(),
            ));
        }
        Ok(BuilderConfig {
            inner: config_ptr as *mut _,
        })
    }

    pub fn build_serialized_network(
        &self,
        network: &mut NetworkDefinition,
        config: &mut BuilderConfig,
    ) -> Result<Vec<u8>> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid builder".to_string()));
        }
        let network_ptr = network.as_mut_ptr();
        let config_ptr = config.as_mut_ptr();

        let serialized_engine = unsafe {
            let builder = &mut *(self.inner as *mut trtx_sys::nvinfer1::IBuilder);
            let network = &mut *(network_ptr as *mut trtx_sys::nvinfer1::INetworkDefinition);
            let config = &mut *(config_ptr as *mut trtx_sys::nvinfer1::IBuilderConfig);
            let mut builder_pin = std::pin::Pin::new_unchecked(builder);
            builder_pin.as_mut().buildSerializedNetwork(
                std::pin::Pin::new_unchecked(network),
                std::pin::Pin::new_unchecked(config),
            )
        };

        if serialized_engine.is_null() {
            return Err(Error::Runtime(
                "Failed to build serialized network".to_string(),
            ));
        }

        let data = unsafe {
            let host_memory = &*serialized_engine;
            let size_raw = host_memory.size();
            let size = size_raw.0 as usize;
            let data_ptr = host_memory.data();
            let slice = std::slice::from_raw_parts(data_ptr as *const u8, size);
            slice.to_vec()
        };

        Ok(data)
    }
}

impl Drop for Builder<'_> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_sys::delete_builder(self.inner);
            }
        }
    }
}

unsafe impl Send for Builder<'_> {}
