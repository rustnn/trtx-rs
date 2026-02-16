//! Real TensorRT builder implementation

use std::pin::Pin;
use std::ptr;

use crate::error::{Error, Result};
use crate::logger::Logger;
use crate::network::NetworkDefinition;
use trtx_sys::nvinfer1::{self, IBuilderConfig, ProfilingVerbosity};

/// Builder configuration (real mode)
pub struct BuilderConfig<'builder> {
    inner: Pin<&'builder mut IBuilderConfig>,
}

impl<'builder> BuilderConfig<'builder> {
    /// See [IBuilderConfig::setMemoryPoolLimit]
    pub fn set_memory_pool_limit(&mut self, pool: nvinfer1::MemoryPoolType, size: usize) {
        self.inner.as_mut().setMemoryPoolLimit(pool, size);
    }

    /// See [IBuilderConfig::setProfilingVerbosity]
    pub fn set_profiling_verbosity(&mut self, verbosity: nvinfer1::ProfilingVerbosity) {
        self.inner.as_mut().setProfilingVerbosity(verbosity);
    }

    /// See [IBuilderConfig::getProfilingVerbosity]
    pub fn get_profiling_verbosity(&self) -> ProfilingVerbosity {
        self.inner.as_ref().getProfilingVerbosity()
    }

    /// See [IBuilderConfig::setAvgTimingIterations]
    pub fn set_avg_timing_iterations(&mut self, avg_timing: i32) {
        self.inner.as_mut().setAvgTimingIterations(avg_timing);
    }

    /// See [IBuilderConfig::getAvgTimingIterations]
    pub fn get_avg_timing_iterations(&self) -> i32 {
        self.inner.as_ref().getAvgTimingIterations()
    }

    /// See [IBuilderConfig::setEngineCapability]
    pub fn set_engine_capability(&mut self, capability: nvinfer1::EngineCapability) {
        self.inner.as_mut().setEngineCapability(capability);
    }

    /// See [IBuilderConfig::getEngineCapability]
    pub fn get_engine_capability(&self) -> nvinfer1::EngineCapability {
        self.inner.as_ref().getEngineCapability()
    }

    /// See [IBuilderConfig::setFlags]
    pub fn set_flags(&mut self, flags: nvinfer1::BuilderFlags) {
        self.inner.as_mut().setFlags(flags);
    }

    /// See [IBuilderConfig::getFlags]
    pub fn get_flags(&self) -> nvinfer1::BuilderFlags {
        self.inner.as_ref().getFlags()
    }

    /// See [IBuilderConfig::setFlag]
    pub fn set_flag(&mut self, flag: nvinfer1::BuilderFlag) {
        self.inner.as_mut().setFlag(flag);
    }

    /// See [IBuilderConfig::clearFlag]
    pub fn clear_flag(&mut self, flag: nvinfer1::BuilderFlag) {
        self.inner.as_mut().clearFlag(flag);
    }

    /// See [IBuilderConfig::getFlag]
    pub fn get_flag(&self, flag: nvinfer1::BuilderFlag) -> bool {
        self.inner.as_ref().getFlag(flag)
    }

    /// See [IBuilderConfig::setDLACore]
    pub fn set_dla_core(&mut self, dla_core: i32) {
        self.inner.as_mut().setDLACore(dla_core);
    }

    /// See [IBuilderConfig::getDLACore]
    pub fn get_dla_core(&self) -> i32 {
        self.inner.as_ref().getDLACore()
    }

    /// See [IBuilderConfig::setDefaultDeviceType]
    pub fn set_default_device_type(&mut self, device_type: nvinfer1::DeviceType) {
        self.inner.as_mut().setDefaultDeviceType(device_type);
    }

    /// See [IBuilderConfig::getDefaultDeviceType]
    pub fn get_default_device_type(&self) -> nvinfer1::DeviceType {
        self.inner.as_ref().getDefaultDeviceType()
    }

    /// See [IBuilderConfig::reset]
    pub fn reset(&mut self) {
        self.inner.as_mut().reset();
    }

    /// See [IBuilderConfig::getNbOptimizationProfiles]
    pub fn get_nb_optimization_profiles(&self) -> i32 {
        self.inner.as_ref().getNbOptimizationProfiles()
    }

    /// See [IBuilderConfig::setTacticSources]
    pub fn set_tactic_sources(&mut self, sources: nvinfer1::TacticSources) -> bool {
        self.inner.as_mut().setTacticSources(sources)
    }

    /// See [IBuilderConfig::getTacticSources]
    pub fn get_tactic_sources(&self) -> nvinfer1::TacticSources {
        self.inner.as_ref().getTacticSources()
    }

    /// See [IBuilderConfig::getMemoryPoolLimit]
    pub fn get_memory_pool_limit(&self, pool: nvinfer1::MemoryPoolType) -> usize {
        self.inner.as_ref().getMemoryPoolLimit(pool)
    }

    /// See [IBuilderConfig::setPreviewFeature]
    pub fn set_preview_feature(&mut self, feature: nvinfer1::PreviewFeature, enable: bool) {
        self.inner.as_mut().setPreviewFeature(feature, enable);
    }

    /// See [IBuilderConfig::getPreviewFeature]
    pub fn get_preview_feature(&self, feature: nvinfer1::PreviewFeature) -> bool {
        self.inner.as_ref().getPreviewFeature(feature)
    }

    /// See [IBuilderConfig::setBuilderOptimizationLevel]
    pub fn set_builder_optimization_level(&mut self, level: i32) {
        self.inner.as_mut().setBuilderOptimizationLevel(level);
    }

    /// See [IBuilderConfig::getBuilderOptimizationLevel]
    pub fn get_builder_optimization_level(&mut self) -> i32 {
        self.inner.as_mut().getBuilderOptimizationLevel()
    }

    /// See [IBuilderConfig::setHardwareCompatibilityLevel]
    pub fn set_hardware_compatibility_level(
        &mut self,
        level: nvinfer1::HardwareCompatibilityLevel,
    ) {
        self.inner.as_mut().setHardwareCompatibilityLevel(level);
    }

    /// See [IBuilderConfig::getHardwareCompatibilityLevel]
    pub fn get_hardware_compatibility_level(&self) -> nvinfer1::HardwareCompatibilityLevel {
        self.inner.as_ref().getHardwareCompatibilityLevel()
    }

    /// See [IBuilderConfig::setMaxAuxStreams]
    pub fn set_max_aux_streams(&mut self, nb_streams: i32) {
        self.inner.as_mut().setMaxAuxStreams(nb_streams);
    }

    /// See [IBuilderConfig::getMaxAuxStreams]
    pub fn get_max_aux_streams(&self) -> i32 {
        self.inner.as_ref().getMaxAuxStreams()
    }

    /// See [IBuilderConfig::setRuntimePlatform]
    pub fn set_runtime_platform(&mut self, platform: nvinfer1::RuntimePlatform) {
        self.inner.as_mut().setRuntimePlatform(platform);
    }

    /// See [IBuilderConfig::getRuntimePlatform]
    pub fn get_runtime_platform(&self) -> nvinfer1::RuntimePlatform {
        self.inner.as_ref().getRuntimePlatform()
    }

    /// See [IBuilderConfig::setMaxNbTactics]
    pub fn set_max_nb_tactics(&mut self, max_nb_tactics: i32) {
        self.inner.as_mut().setMaxNbTactics(max_nb_tactics);
    }

    /// See [IBuilderConfig::getMaxNbTactics]
    pub fn get_max_nb_tactics(&self) -> i32 {
        self.inner.as_ref().getMaxNbTactics()
    }

    /// See [IBuilderConfig::setTilingOptimizationLevel]
    pub fn set_tiling_optimization_level(
        &mut self,
        level: nvinfer1::TilingOptimizationLevel,
    ) -> bool {
        self.inner.as_mut().setTilingOptimizationLevel(level)
    }

    /// See [IBuilderConfig::getTilingOptimizationLevel]
    pub fn get_tiling_optimization_level(&self) -> nvinfer1::TilingOptimizationLevel {
        self.inner.as_ref().getTilingOptimizationLevel()
    }

    /// See [IBuilderConfig::setL2LimitForTiling]
    pub fn set_l2_limit_for_tiling(&mut self, size: i64) -> bool {
        self.inner.as_mut().setL2LimitForTiling(size)
    }

    /// See [IBuilderConfig::getL2LimitForTiling]
    pub fn get_l2_limit_for_tiling(&self) -> i64 {
        self.inner.as_ref().getL2LimitForTiling()
    }

    /// See [IBuilderConfig::setNbComputeCapabilities]
    pub fn set_nb_compute_capabilities(&mut self, max_nb_compute_capabilities: i32) -> bool {
        self.inner
            .as_mut()
            .setNbComputeCapabilities(max_nb_compute_capabilities)
    }

    /// See [IBuilderConfig::getNbComputeCapabilities]
    pub fn get_nb_compute_capabilities(&self) -> i32 {
        self.inner.as_ref().getNbComputeCapabilities()
    }

    /// See [IBuilderConfig::setComputeCapability]
    pub fn set_compute_capability(
        &mut self,
        compute_capability: nvinfer1::ComputeCapability,
        index: i32,
    ) -> bool {
        self.inner
            .as_mut()
            .setComputeCapability(compute_capability, index)
    }

    /// See [IBuilderConfig::getComputeCapability]
    pub fn get_compute_capability(&self, index: i32) -> nvinfer1::ComputeCapability {
        self.inner.as_ref().getComputeCapability(index)
    }
}

impl Drop for BuilderConfig<'_> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(self.inner.as_mut().get_unchecked_mut());
        }
    }
}

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

    pub fn build_serialized_network<'network, 'config, 'config_borrow>(
        &'builder self,
        network: &'network mut NetworkDefinition,
        config: &'config_borrow mut BuilderConfig<'config>,
    ) -> Result<Vec<u8>> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid builder".to_string()));
        }
        let network_ptr = network.as_mut_ptr();

        let serialized_engine = unsafe {
            let builder = &mut *(self.inner as *mut trtx_sys::nvinfer1::IBuilder);
            let network = &mut *(network_ptr as *mut trtx_sys::nvinfer1::INetworkDefinition);
            let mut builder_pin = std::pin::Pin::new_unchecked(builder);
            builder_pin.as_mut().buildSerializedNetwork(
                std::pin::Pin::new_unchecked(network),
                config.inner.as_mut(),
            )
        };

        if serialized_engine.is_null() {
            return Err(Error::Runtime(
                "Failed to build serialized network".to_string(),
            ));
        }

        let data = unsafe {
            let host_memory = serialized_engine.as_mut().unwrap();
            let size = host_memory.size();
            let data_ptr = host_memory.data();
            let slice = std::slice::from_raw_parts(data_ptr as *const u8, size);
            ptr::drop_in_place(host_memory);
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
