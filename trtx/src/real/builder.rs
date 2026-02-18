//! Real TensorRT builder implementation
use crate::error::{Error, Result};
use crate::logger::Logger;
use crate::network::NetworkDefinition;
use crate::real::host_memory::HostMemory;

pub use super::builder_config::BuilderConfig;

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

    pub fn create_network(&self, flags: u32) -> Result<NetworkDefinition<'_>> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid builder".to_string()));
        }
        let network_ptr = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::IBuilder>(self.inner)
                .createNetworkV2(flags)
        };
        let network = unsafe { network_ptr.as_mut() }
            .ok_or_else(|| Error::Runtime("Failed to create network".to_string()))?;
        Ok(NetworkDefinition::from_ptr(network))
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

    pub fn build_serialized_network<'network, 'config, 'config_borrow>(
        &self,
        network: &'network NetworkDefinition,
        config: &'config_borrow mut BuilderConfig<'config>,
    ) -> Result<HostMemory<'_>> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid builder".to_string()));
        }

        let serialized_engine = unsafe {
            let builder = &mut *(self.inner as *mut trtx_sys::nvinfer1::IBuilder);
            let mut builder_pin = std::pin::Pin::new_unchecked(builder);
            builder_pin
                .as_mut()
                .buildSerializedNetwork(network.inner.lock()?.as_mut(), config.inner.as_mut())
                .as_mut()
        }
        .ok_or_else(|| Error::Runtime("Failed to build serialized network".to_string()))?;

        Ok(unsafe { HostMemory::from_raw_ref(self, serialized_engine) })
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
