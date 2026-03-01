//! Real TensorRT builder implementation
use crate::error::{Error, Result};
use crate::logger::Logger;
use crate::network::NetworkDefinition;
use crate::real::host_memory::HostMemory;
use autocxx::cxx::memory::UniquePtr;
use trtx_sys::nvinfer1::IBuilder;

pub use super::builder_config::BuilderConfig;

/// Builder (real mode)
pub struct Builder<'a> {
    inner: UniquePtr<IBuilder>,
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
        #[cfg(not(feature = "mock"))]
        {
            use trtx_sys::nvinfer1::IBuilder;

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
            } as *mut IBuilder;
            if builder_ptr.is_null() {
                return Err(Error::Runtime("Failed to create builder".to_string()));
            }
            Ok(Builder {
                inner: unsafe { UniquePtr::from_raw(builder_ptr) },
                _logger: logger,
            })
        }
        #[cfg(feature = "mock")]
        Ok(Builder {
            inner: UniquePtr::null(),
            _logger: logger,
        })
    }

    pub fn create_network(&'_ mut self, flags: u32) -> Result<NetworkDefinition<'builder>> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid builder".to_string()));
        }
        let network_ptr = self.inner.pin_mut().createNetworkV2(flags);
        let network = unsafe { network_ptr.as_mut() }
            .ok_or_else(|| Error::Runtime("Failed to create network".to_string()))?;
        Ok(NetworkDefinition::from_ptr(network))
    }

    pub fn create_config(&'_ mut self) -> Result<BuilderConfig> {
        #[cfg(not(feature = "mock"))]
        let config_ptr = self.inner.pin_mut().createBuilderConfig();
        #[cfg(feature = "mock")]
        let config_ptr = std::ptr::null_mut();
        BuilderConfig::new(config_ptr)
    }

    pub fn build_serialized_network<'config_borrow, 'output>(
        &mut self,
        network: &mut NetworkDefinition,
        config: &'config_borrow mut BuilderConfig,
    ) -> Result<HostMemory<'output>>
    where
        'output: 'config_borrow + 'builder,
    {
        let serialized_engine = unsafe {
            self.inner
                .pin_mut()
                .buildSerializedNetwork(network.inner.pin_mut(), config.inner.pin_mut())
                .as_mut()
        }
        .ok_or_else(|| Error::Runtime("Failed to build serialized network".to_string()))?;

        Ok(unsafe { HostMemory::from_raw(serialized_engine) })
    }
}
