//! Builder for creating TensorRT engines

use crate::error::{Error, Result};
use crate::logger::Logger;
use crate::network::NetworkDefinition;

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

/// Builder configuration
pub struct BuilderConfig {
    #[cfg(not(feature = "mock"))]
    inner: *mut std::ffi::c_void,
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxBuilderConfig,
}

impl BuilderConfig {
    /// Set memory pool limit
    pub fn set_memory_pool_limit(&mut self, pool: MemoryPoolType, size: usize) -> Result<()> {
        #[cfg(feature = "mock")]
        {
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_builder_config_set_memory_pool_limit(
                    self.inner,
                    pool as i32,
                    size,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(())
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid builder config".to_string()));
            }

            let trt_pool = match pool {
                MemoryPoolType::Workspace => 0,      // kWORKSPACE
                MemoryPoolType::DlaManagedSram => 1, // kDLA_MANAGED_SRAM
                MemoryPoolType::DlaLocalDram => 2,   // kDLA_LOCAL_DRAM
                MemoryPoolType::DlaGlobalDram => 3,  // kDLA_GLOBAL_DRAM
            };

            unsafe {
                trtx_sys::builder_config_set_memory_pool_limit(self.inner, trt_pool, size);
            }

            Ok(())
        }
    }

    /// Get the raw pointer (for internal use)
    #[cfg(not(feature = "mock"))]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.inner
    }

    #[cfg(feature = "mock")]
    pub(crate) fn as_ptr(&self) -> *mut trtx_sys::TrtxBuilderConfig {
        self.inner
    }

    #[cfg(feature = "mock")]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut trtx_sys::TrtxBuilderConfig {
        self.inner
    }
}

impl Drop for BuilderConfig {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            #[cfg(feature = "mock")]
            unsafe {
                trtx_sys::trtx_builder_config_destroy(self.inner);
            }
            #[cfg(not(feature = "mock"))]
            unsafe {
                trtx_sys::delete_config(self.inner);
            }
        }
    }
}

unsafe impl Send for BuilderConfig {}

/// Builder for creating optimized TensorRT engines
pub struct Builder<'a> {
    #[cfg(not(feature = "mock"))]
    inner: *mut std::ffi::c_void,
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxBuilder,
    _logger: &'a Logger,
}

impl<'a> Builder<'a> {
    /// Create a new builder
    pub fn new(logger: &'a Logger) -> Result<Self> {
        #[cfg(feature = "mock")]
        {
            let mut builder_ptr: *mut trtx_sys::TrtxBuilder = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_builder_create(
                    logger.as_ptr(),
                    &mut builder_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(Builder {
                inner: builder_ptr,
                _logger: logger,
            })
        }

        #[cfg(not(feature = "mock"))]
        {
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
    }

    /// Create a network definition
    pub fn create_network(&self, flags: u32) -> Result<NetworkDefinition> {
        #[cfg(feature = "mock")]
        {
            let mut network_ptr: *mut trtx_sys::TrtxNetworkDefinition = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_builder_create_network(
                    self.inner,
                    flags,
                    &mut network_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(NetworkDefinition::from_ptr(network_ptr))
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid builder".to_string()));
            }

            // Use autocxx Pin to call createNetworkV2 directly (no C++ wrapper!)
            let network_ptr = unsafe {
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::IBuilder>(self.inner)
                    .createNetworkV2(flags)
            };

            if network_ptr.is_null() {
                return Err(Error::Runtime("Failed to create network".to_string()));
            }

            Ok(NetworkDefinition::from_ptr(network_ptr as *mut _))
        }
    }

    /// Create a builder configuration
    pub fn create_config(&self) -> Result<BuilderConfig> {
        #[cfg(feature = "mock")]
        {
            let mut config_ptr: *mut trtx_sys::TrtxBuilderConfig = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_builder_create_builder_config(
                    self.inner,
                    &mut config_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(BuilderConfig { inner: config_ptr })
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid builder".to_string()));
            }

            // Use autocxx Pin to call createBuilderConfig directly
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
    }

    /// Build a serialized network (engine)
    pub fn build_serialized_network(
        &self,
        network: &mut NetworkDefinition,
        config: &mut BuilderConfig,
    ) -> Result<Vec<u8>> {
        #[cfg(feature = "mock")]
        {
            let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut size: usize = 0;
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_builder_build_serialized_network(
                    self.inner,
                    network.as_ptr(),
                    config.as_ptr(),
                    &mut data_ptr,
                    &mut size,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
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

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid builder".to_string()));
            }
            let network_ptr = network.as_mut_ptr();
            let config_ptr = config.as_mut_ptr();

            // Use autocxx Pin to call buildSerializedNetwork directly
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

            // Get data from IHostMemory and copy to Vec
            let data = unsafe {
                let host_memory = &*serialized_engine;
                // autocxx wraps types, access the inner value
                let size_raw = host_memory.size();
                let size = size_raw.0 as usize; // Access inner value and convert
                let data_ptr = host_memory.data();
                let slice = std::slice::from_raw_parts(data_ptr as *const u8, size);
                let vec = slice.to_vec();
                // IHostMemory will be freed when it goes out of scope
                vec
            };

            Ok(data)
        }
    }
}

impl Drop for Builder<'_> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            #[cfg(feature = "mock")]
            unsafe {
                trtx_sys::trtx_builder_destroy(self.inner);
            }
            #[cfg(not(feature = "mock"))]
            unsafe {
                trtx_sys::delete_builder(self.inner);
            }
        }
    }
}

unsafe impl Send for Builder<'_> {}
