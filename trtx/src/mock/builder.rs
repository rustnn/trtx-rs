//! Mock builder implementations

use crate::error::Result;
use crate::logger::Logger;
use crate::network::NetworkDefinition;
use trtx_sys::{
    BuilderFlag, ComputeCapability, DeviceType, EngineCapability, HardwareCompatibilityLevel,
    MemoryPoolType, PreviewFeature, ProfilingVerbosity, RuntimePlatform, TilingOptimizationLevel,
};

/// Builder configuration (mock mode)
pub struct BuilderConfig {
    pub(crate) inner: *mut trtx_sys::TrtxBuilderConfig,
}

impl BuilderConfig {
    pub fn set_memory_pool_limit(&mut self, pool: MemoryPoolType, size: usize) {
        set_memory_pool_limit(self.inner, pool as i32, size)
    }

    pub(crate) fn as_ptr(&self) -> *mut trtx_sys::TrtxBuilderConfig {
        self.inner
    }

    pub fn set_profiling_verbosity(&mut self, _verbosity: ProfilingVerbosity) {}

    pub fn get_profiling_verbosity(&self) -> ProfilingVerbosity {
        ProfilingVerbosity::kDETAILED
    }

    pub fn set_avg_timing_iterations(&mut self, _avg_timing: i32) {}

    pub fn get_avg_timing_iterations(&self) -> i32 {
        1
    }

    pub fn set_engine_capability(&mut self, _capability: EngineCapability) {}

    pub fn get_engine_capability(&self) -> EngineCapability {
        EngineCapability::kSTANDARD
    }

    pub fn set_flags(&mut self, _flags: u32) {}

    pub fn get_flags(&self) -> u32 {
        0
    }

    pub fn set_flag(&mut self, _flag: BuilderFlag) {}

    pub fn clear_flag(&mut self, _flag: BuilderFlag) {}

    pub fn get_flag(&self, _flag: BuilderFlag) -> bool {
        false
    }

    pub fn set_dla_core(&mut self, _dla_core: i32) {}

    pub fn get_dla_core(&self) -> i32 {
        -1
    }

    pub fn set_default_device_type(&mut self, _device_type: DeviceType) {}

    pub fn get_default_device_type(&self) -> DeviceType {
        DeviceType::kGPU
    }

    pub fn reset(&mut self) {}

    pub fn get_nb_optimization_profiles(&self) -> i32 {
        0
    }

    pub fn set_tactic_sources(&mut self, _sources: u32) -> bool {
        true
    }

    pub fn get_tactic_sources(&self) -> u32 {
        0
    }

    pub fn get_memory_pool_limit(&self, _pool: MemoryPoolType) -> usize {
        0
    }

    pub fn set_preview_feature(&mut self, _feature: PreviewFeature, _enable: bool) {}

    pub fn get_preview_feature(&self, _feature: PreviewFeature) -> bool {
        false
    }

    pub fn set_builder_optimization_level(&mut self, _level: i32) {}

    pub fn get_builder_optimization_level(&mut self) -> i32 {
        3
    }

    pub fn set_hardware_compatibility_level(&mut self, _level: HardwareCompatibilityLevel) {}

    pub fn get_hardware_compatibility_level(&self) -> HardwareCompatibilityLevel {
        HardwareCompatibilityLevel::kNONE
    }

    pub fn set_max_aux_streams(&mut self, _nb_streams: i32) {}

    pub fn get_max_aux_streams(&self) -> i32 {
        0
    }

    pub fn set_runtime_platform(&mut self, _platform: RuntimePlatform) {}

    pub fn get_runtime_platform(&self) -> RuntimePlatform {
        RuntimePlatform::kSAME_AS_BUILD
    }

    pub fn set_max_nb_tactics(&mut self, _max_nb_tactics: i32) {}

    pub fn get_max_nb_tactics(&self) -> i32 {
        -1
    }

    pub fn set_tiling_optimization_level(&mut self, _level: TilingOptimizationLevel) -> bool {
        true
    }

    pub fn get_tiling_optimization_level(&self) -> TilingOptimizationLevel {
        TilingOptimizationLevel::kNONE
    }

    pub fn set_l2_limit_for_tiling(&mut self, _size: i64) -> bool {
        true
    }

    pub fn get_l2_limit_for_tiling(&self) -> i64 {
        0
    }

    pub fn set_nb_compute_capabilities(&mut self, _max_nb_compute_capabilities: i32) -> bool {
        true
    }

    pub fn get_nb_compute_capabilities(&self) -> i32 {
        0
    }

    pub fn set_compute_capability(
        &mut self,
        _compute_capability: ComputeCapability,
        _index: i32,
    ) -> bool {
        true
    }

    pub fn get_compute_capability(&self, _index: i32) -> ComputeCapability {
        ComputeCapability::kNONE
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
pub(crate) fn trtx_builder_create(
    logger_ptr: *mut trtx_sys::TrtxLogger,
) -> Result<*mut trtx_sys::TrtxBuilder> {
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

fn set_memory_pool_limit(config_ptr: *mut trtx_sys::TrtxBuilderConfig, pool: i32, size: usize) {
    let mut error_msg = [0i8; 1024];

    unsafe {
        trtx_sys::trtx_builder_config_set_memory_pool_limit(
            config_ptr,
            pool,
            size,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };
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

    Ok(NetworkDefinition::from_ptr(
        network_ptr as *mut std::ffi::c_void,
    ))
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
