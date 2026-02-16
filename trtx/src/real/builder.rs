//! Real TensorRT builder implementation

use std::pin::Pin;

use crate::error::{Error, Result};
use crate::logger::Logger;
use crate::network::NetworkDefinition;
use trtx_sys::nvinfer1::{self, ProfilingVerbosity};

/// Builder configuration (real mode)
pub struct BuilderConfig<'builder> {
    inner: Pin<&'builder mut nvinfer1::IBuilderConfig>,
}

impl<'builder> BuilderConfig<'builder> {
    pub fn set_memory_pool_limit(
        &mut self,
        pool: nvinfer1::MemoryPoolType,
        size: usize,
    ) -> Result<()> {
        self.inner.as_mut().setMemoryPoolLimit(pool, size);
        Ok(())
    }

    pub fn set_profiling_verbosity(
        &mut self,
        verbosity: nvinfer1::ProfilingVerbosity,
    ) -> Result<()> {
        self.inner.as_mut().setProfilingVerbosity(verbosity);
        Ok(())
    }

    pub fn get_profiling_verbosity(&self) -> ProfilingVerbosity {
        self.inner.as_ref().getProfilingVerbosity()
    }

    pub(crate) fn as_mut(&'builder mut self) -> Pin<&'builder mut nvinfer1::IBuilderConfig> {
        self.inner.as_mut()
    }
}

unsafe impl Send for BuilderConfig<'_> {}

/// Builder (real mode)
pub struct Builder<'a> {
    inner: *mut std::ffi::c_void,
    _logger: &'a Logger,
}

impl<'builder> Builder<'builder> {
    #[cfg(not(feature = "link_tensorrt_rtx"))]
    #[cfg(not(feature = "dlopen_tensorrt_rtx"))]
    pub fn new(logger: &'a Logger) -> Result<Self> {
        Err(Error::TrtRtxLibraryNotLoaded)
    }

    #[cfg(any(feature = "link_tensorrt_rtx", feature = "dlopen_tensorrt_rtx"))]
    pub fn new(logger: &'builder Logger) -> Result<Self> {
        let logger_ptr = logger.as_logger_ptr();
        let builder_ptr = {
            #[cfg(feature = "link_tensorrt_rtx")]
            unsafe {
                trtx_sys::create_infer_builder(logger_ptr)
            }
            #[cfg(not(feature = "link_tensorrt_rtx"))]
            #[cfg(feature = "dlopen_tensorrt_rtx")]
            unsafe {
                use libloading::Symbol;
                use std::ffi::c_void;

                use crate::TRTLIB;
                if !TRTLIB.read()?.is_some() {
                    crate::dynamically_load_tensorrt(None::<String>)?;
                }

                let lock = TRTLIB.read()?;
                let create_infer_builder: Symbol<fn(*mut c_void, u32) -> *mut c_void> = lock
                    .as_ref()
                    .ok_or(Error::TrtRtxLibraryNotLoaded)?
                    .get(b"createInferBuilder_INTERNAL")?;
                create_infer_builder(logger_ptr, trtx_sys::get_tensorrt_version())
            }
        };
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

    pub fn create_config(&self) -> Result<BuilderConfig<'builder>> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid builder".to_string()));
        }
        unsafe {
            let config_ptr =
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::IBuilder>(self.inner)
                    .createBuilderConfig()
                    .as_mut()
                    .ok_or_else(|| Error::Runtime("Failed to create builder config".to_string()))?;
            Ok(BuilderConfig {
                inner: std::pin::Pin::new_unchecked(config_ptr),
            })
        }
    }

    pub fn build_serialized_network<'config>(
        &self,
        network: &mut NetworkDefinition,
        config: &'config mut BuilderConfig<'config>,
    ) -> Result<Vec<u8>> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid builder".to_string()));
        }
        let network_ptr = network.as_mut_ptr();

        let serialized_engine = unsafe {
            let builder = &mut *(self.inner as *mut trtx_sys::nvinfer1::IBuilder);
            let network = &mut *(network_ptr as *mut trtx_sys::nvinfer1::INetworkDefinition);
            let mut builder_pin = std::pin::Pin::new_unchecked(builder);
            builder_pin
                .as_mut()
                .buildSerializedNetwork(std::pin::Pin::new_unchecked(network), config.as_mut())
        };

        if serialized_engine.is_null() {
            return Err(Error::Runtime(
                "Failed to build serialized network".to_string(),
            ));
        }

        let data = unsafe {
            let host_memory = &*serialized_engine;
            let size = host_memory.size();
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
