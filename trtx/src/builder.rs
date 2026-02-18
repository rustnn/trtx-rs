//! Builder for creating TensorRT engines

use crate::error::{Error, Result};
use crate::host_memory::HostMemory;
use crate::interfaces::{ErrorRecorder, RecordError};
use crate::logger::Logger;
use crate::network::NetworkDefinition;
use crate::optimization_profile::OptimizationProfile;
use autocxx::cxx::memory::UniquePtr;
use std::marker::PhantomData;
use std::pin::Pin;
use trtx_sys::nvinfer1::IBuilder;

/// Network definition builder flags
pub mod network_flags {
    /// Explicit batch sizes
    pub const EXPLICIT_BATCH: u32 = 1 << 0;
}

pub use crate::builder_config::BuilderConfig;
pub use trtx_sys::{
    BuilderFlag, ComputeCapability, DeviceType, EngineCapability, HardwareCompatibilityLevel,
    MemoryPoolType, PreviewFeature, ProfilingVerbosity, RuntimePlatform, TilingOptimizationLevel,
};

/// Builder for creating TensorRT engines
pub struct Builder<'a> {
    inner: UniquePtr<IBuilder>,
    _logger: PhantomData<&'a Logger>,
    error_recorder: Option<Pin<Box<ErrorRecorder>>>,
}

impl<'builder> Builder<'builder> {
    #[cfg(not(feature = "link_tensorrt_rtx"))]
    #[cfg(not(feature = "dlopen_tensorrt_rtx"))]
    pub fn new(logger: &'builder Logger) -> Result<Self> {
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
                    let create_infer_builder: Symbol<fn(*mut c_void, u32) -> *mut IBuilder> = lock
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
                inner: unsafe { UniquePtr::from_raw(builder_ptr) },
                error_recorder: None,
                _logger: Default::default(),
            })
        }
        #[cfg(feature = "mock")]
        Ok(Builder {
            inner: UniquePtr::null(),
            _logger: Default::default(),
            error_recorder: None,
        })
    }

    pub fn create_network(&'_ mut self, flags: u32) -> Result<NetworkDefinition<'builder>> {
        if cfg!(feature = "mock") {
            Ok(NetworkDefinition::from_ptr(std::ptr::null_mut()))
        } else {
            let network_ptr = self.inner.pin_mut().createNetworkV2(flags);
            let network = unsafe { network_ptr.as_mut() }
                .ok_or_else(|| Error::Runtime("Failed to create network".to_string()))?;
            Ok(NetworkDefinition::from_ptr(network))
        }
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
        if cfg!(feature = "mock") {
            Ok(unsafe { HostMemory::from_raw(std::ptr::null_mut()) })
        } else {
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

    pub fn creata_optimization_profile(&mut self) -> Result<OptimizationProfile<'builder>> {
        let profile = unsafe {
            self.inner
                .pin_mut()
                .createOptimizationProfile()
                .as_mut()
                .ok_or_else(|| {
                    Error::Runtime("Failed to create optimization profile".to_string())
                })?
        };
        Ok(OptimizationProfile::from_raw(profile))
    }

    /// See [trtx_sys::nvinfer1::IBuilder::setErrorRecorder]
    ///
    /// The Rust bindings only allow setting the error recorder once
    pub fn set_error_recorder(&mut self, error_recorder: Box<dyn RecordError>) -> Result<()> {
        let error_recorder = ErrorRecorder::new(error_recorder)?;
        if self.error_recorder.is_some() {
            // would need to make sure that we don't destroy a monitor still in use
            // could offer this as an unsafe method for users who only set this when there is no
            // build process active. Or we only accept a ref to progress monitor and force user
            // via lifetimes to keep this alive for builder config lifetime
            panic!("Setting a progress monitor more than once not supported at the moment");
        }
        self.error_recorder = Some(error_recorder);
        let rec = self
            .error_recorder
            .as_mut()
            .unwrap()
            .as_trt_error_recorder();
        #[cfg(not(feature = "mock"))]
        unsafe {
            self.inner.pin_mut().setErrorRecorder(rec)
        };
        Ok(())
    }
}
