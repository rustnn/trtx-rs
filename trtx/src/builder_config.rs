//! Real TensorRT builder config implementation

use std::pin::Pin;

use crate::error::PropertySetAttempt;
use crate::interfaces::ProgressMonitor;
use crate::Error;
use crate::Result;
use cxx::UniquePtr;
use trtx_sys::nvinfer1::{self, IBuilderConfig};
use trtx_sys::{
    BuilderFlag, ComputeCapability, DeviceType, EngineCapability, HardwareCompatibilityLevel,
    MemoryPoolType, PreviewFeature, ProfilingVerbosity, RuntimePlatform, TilingOptimizationLevel,
};

/// Builder configuration (real mode)
pub struct BuilderConfig {
    pub(crate) inner: UniquePtr<IBuilderConfig>,
    progress_monitor: Option<Pin<Box<ProgressMonitor>>>,
}

impl BuilderConfig {
    pub(crate) fn new(builder_config: *mut nvinfer1::IBuilderConfig) -> Result<Self> {
        #[cfg(not(feature = "mock"))]
        if builder_config.is_null() {
            return Err(Error::BuilderConfigCreationFailed);
        }
        Ok(Self {
            inner: unsafe { UniquePtr::from_raw(builder_config) },
            progress_monitor: None,
        })
    }

    /// See [IBuilderConfig::setProgressMonitor]
    /// The Rust bindings only allow setting the progress monitor once per builder config object
    pub fn set_progress_monitor(&mut self, progress_monitor: Pin<Box<ProgressMonitor>>) {
        if self.progress_monitor.is_some() {
            // would need to make sure that we don't destroy a monitor still in use
            // could offer this as an unsafe method for users who only set this when there is no
            // build process active. Or we only accept a ref to progress monitor and force user
            // via lifetimes to keep this alive for builder config lifetime
            panic!("Setting a progress monitor more than once not supported at the moment");
        }
        self.progress_monitor = Some(progress_monitor);
        #[cfg(not(feature = "mock"))]
        unsafe {
            self.inner.pin_mut().setProgressMonitor(
                self.progress_monitor
                    .as_mut()
                    .expect("progress_monitor can't be empty. we just set it")
                    .as_trt_progress_monitor(),
            )
        };
    }

    /// See [IBuilderConfig::setMemoryPoolLimit]
    pub fn set_memory_pool_limit(&mut self, pool: MemoryPoolType, size: usize) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setMemoryPoolLimit(pool.into(), size);
    }

    /// See [IBuilderConfig::setProfilingVerbosity]
    pub fn set_profiling_verbosity(&mut self, verbosity: ProfilingVerbosity) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setProfilingVerbosity(verbosity.into());
    }

    /// See [IBuilderConfig::getProfilingVerbosity]
    pub fn get_profiling_verbosity(&self) -> ProfilingVerbosity {
        if cfg!(not(feature = "mock")) {
            self.inner.getProfilingVerbosity().into()
        } else {
            ProfilingVerbosity::kNONE
        }
    }

    /// See [IBuilderConfig::setAvgTimingIterations]
    pub fn set_avg_timing_iterations(&mut self, avg_timing: i32) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setAvgTimingIterations(avg_timing);
    }

    /// See [IBuilderConfig::getAvgTimingIterations]
    pub fn get_avg_timing_iterations(&self) -> i32 {
        if cfg!(not(feature = "mock")) {
            self.inner.getAvgTimingIterations()
        } else {
            0
        }
    }

    /// See [IBuilderConfig::setEngineCapability]
    pub fn set_engine_capability(&mut self, capability: EngineCapability) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setEngineCapability(capability.into());
    }

    /// See [IBuilderConfig::getEngineCapability]
    pub fn get_engine_capability(&self) -> EngineCapability {
        if cfg!(not(feature = "mock")) {
            self.inner.getEngineCapability().into()
        } else {
            EngineCapability::kSTANDARD
        }
    }

    /// See [IBuilderConfig::setFlags]
    pub fn set_flags(&mut self, flags: u32) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setFlags(flags);
    }

    /// See [IBuilderConfig::getFlags]
    pub fn get_flags(&self) -> u32 {
        if cfg!(not(feature = "mock")) {
            self.inner.getFlags()
        } else {
            0
        }
    }

    /// See [IBuilderConfig::setFlag]
    pub fn set_flag(&mut self, flag: BuilderFlag) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setFlag(flag.into());
    }

    /// See [IBuilderConfig::clearFlag]
    pub fn clear_flag(&mut self, flag: BuilderFlag) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().clearFlag(flag.into());
    }

    /// See [IBuilderConfig::getFlag]
    pub fn get_flag(&self, flag: BuilderFlag) -> bool {
        if cfg!(not(feature = "mock")) {
            self.inner.getFlag(flag.into())
        } else {
            false
        }
    }

    /// See [IBuilderConfig::setDLACore]
    pub fn set_dla_core(&mut self, dla_core: i32) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setDLACore(dla_core);
    }

    /// See [IBuilderConfig::getDLACore]
    pub fn get_dla_core(&self) -> i32 {
        if cfg!(not(feature = "mock")) {
            self.inner.getDLACore()
        } else {
            0
        }
    }

    /// See [IBuilderConfig::setDefaultDeviceType]
    pub fn set_default_device_type(&mut self, device_type: DeviceType) {
        #[cfg(not(feature = "mock"))]
        self.inner
            .pin_mut()
            .setDefaultDeviceType(device_type.into());
    }

    /// See [IBuilderConfig::getDefaultDeviceType]
    pub fn get_default_device_type(&self) -> DeviceType {
        if cfg!(not(feature = "mock")) {
            self.inner.getDefaultDeviceType().into()
        } else {
            DeviceType::kGPU
        }
    }

    /// See [IBuilderConfig::reset]
    pub fn reset(&mut self) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().reset();
    }

    /// See [IBuilderConfig::getNbOptimizationProfiles]
    pub fn get_nb_optimization_profiles(&self) -> i32 {
        if cfg!(not(feature = "mock")) {
            self.inner.getNbOptimizationProfiles()
        } else {
            0
        }
    }

    /// See [IBuilderConfig::setTacticSources]
    pub fn set_tactic_sources(&mut self, sources: u32) -> crate::Result<()> {
        if cfg!(not(feature = "mock")) {
            if self.inner.pin_mut().setTacticSources(sources) {
                Ok(())
            } else {
                Err(crate::Error::FailedToSetProperty(
                    PropertySetAttempt::BuilderConfigTacticSources,
                ))
            }
        } else {
            Ok(())
        }
    }

    /// See [IBuilderConfig::getTacticSources]
    pub fn get_tactic_sources(&self) -> u32 {
        if cfg!(not(feature = "mock")) {
            self.inner.getTacticSources()
        } else {
            0
        }
    }

    /// See [IBuilderConfig::getMemoryPoolLimit]
    pub fn get_memory_pool_limit(&self, pool: MemoryPoolType) -> usize {
        if cfg!(not(feature = "mock")) {
            self.inner.getMemoryPoolLimit(pool.into())
        } else {
            0
        }
    }

    /// See [IBuilderConfig::setPreviewFeature]
    pub fn set_preview_feature(&mut self, feature: PreviewFeature, enable: bool) {
        #[cfg(not(feature = "mock"))]
        self.inner
            .pin_mut()
            .setPreviewFeature(feature.into(), enable);
    }

    /// See [IBuilderConfig::getPreviewFeature]
    pub fn get_preview_feature(&self, feature: PreviewFeature) -> bool {
        if cfg!(not(feature = "mock")) {
            self.inner.getPreviewFeature(feature.into())
        } else {
            false
        }
    }

    /// See [IBuilderConfig::setBuilderOptimizationLevel]
    pub fn set_builder_optimization_level(&mut self, level: i32) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setBuilderOptimizationLevel(level);
    }

    /// See [IBuilderConfig::getBuilderOptimizationLevel]
    pub fn get_builder_optimization_level(&mut self) -> i32 {
        if cfg!(not(feature = "mock")) {
            self.inner.pin_mut().getBuilderOptimizationLevel()
        } else {
            0
        }
    }

    /// See [IBuilderConfig::setHardwareCompatibilityLevel]
    pub fn set_hardware_compatibility_level(&mut self, level: HardwareCompatibilityLevel) {
        #[cfg(not(feature = "mock"))]
        self.inner
            .pin_mut()
            .setHardwareCompatibilityLevel(level.into());
    }

    /// See [IBuilderConfig::getHardwareCompatibilityLevel]
    pub fn get_hardware_compatibility_level(&self) -> HardwareCompatibilityLevel {
        self.inner.getHardwareCompatibilityLevel().into()
    }

    /// See [IBuilderConfig::setMaxAuxStreams]
    pub fn set_max_aux_streams(&mut self, nb_streams: i32) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setMaxAuxStreams(nb_streams);
    }

    /// See [IBuilderConfig::getMaxAuxStreams]
    pub fn get_max_aux_streams(&self) -> i32 {
        if cfg!(not(feature = "mock")) {
            self.inner.getMaxAuxStreams()
        } else {
            0
        }
    }

    /// See [IBuilderConfig::setRuntimePlatform]
    pub fn set_runtime_platform(&mut self, platform: RuntimePlatform) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setRuntimePlatform(platform.into());
    }

    /// See [IBuilderConfig::getRuntimePlatform]
    pub fn get_runtime_platform(&self) -> RuntimePlatform {
        if cfg!(not(feature = "mock")) {
            self.inner.getRuntimePlatform().into()
        } else {
            RuntimePlatform::kSAME_AS_BUILD
        }
    }

    /// See [IBuilderConfig::setMaxNbTactics]
    pub fn set_max_nb_tactics(&mut self, max_nb_tactics: i32) {
        #[cfg(not(feature = "mock"))]
        self.inner.pin_mut().setMaxNbTactics(max_nb_tactics);
    }

    /// See [IBuilderConfig::getMaxNbTactics]
    pub fn get_max_nb_tactics(&self) -> i32 {
        if cfg!(not(feature = "mock")) {
            self.inner.getMaxNbTactics()
        } else {
            0
        }
    }

    /// See [IBuilderConfig::setTilingOptimizationLevel]
    pub fn set_tiling_optimization_level(
        &mut self,
        level: TilingOptimizationLevel,
    ) -> crate::Result<()> {
        if cfg!(not(feature = "mock")) {
            if self
                .inner
                .pin_mut()
                .setTilingOptimizationLevel(level.into())
            {
                Ok(())
            } else {
                Err(crate::Error::FailedToSetProperty(
                    PropertySetAttempt::BuilderConfigTilingOptimizationLevel,
                ))
            }
        } else {
            Ok(())
        }
    }

    /// See [IBuilderConfig::getTilingOptimizationLevel]
    pub fn get_tiling_optimization_level(&self) -> TilingOptimizationLevel {
        if cfg!(not(feature = "mock")) {
            self.inner.getTilingOptimizationLevel().into()
        } else {
            TilingOptimizationLevel::kNONE
        }
    }

    /// See [IBuilderConfig::setL2LimitForTiling]
    pub fn set_l2_limit_for_tiling(&mut self, size: i64) -> crate::Result<()> {
        if cfg!(not(feature = "mock")) {
            if self.inner.pin_mut().setL2LimitForTiling(size) {
                Ok(())
            } else {
                Err(crate::Error::FailedToSetProperty(
                    PropertySetAttempt::BuilderConfigL2LimitForTiling,
                ))
            }
        } else {
            Ok(())
        }
    }

    /// See [IBuilderConfig::getL2LimitForTiling]
    pub fn get_l2_limit_for_tiling(&self) -> i64 {
        if cfg!(not(feature = "mock")) {
            self.inner.getL2LimitForTiling()
        } else {
            0
        }
    }

    /// See [IBuilderConfig::setNbComputeCapabilities]
    pub fn set_nb_compute_capabilities(
        &mut self,
        max_nb_compute_capabilities: i32,
    ) -> crate::Result<()> {
        if cfg!(not(feature = "mock")) {
            if self
                .inner
                .pin_mut()
                .setNbComputeCapabilities(max_nb_compute_capabilities)
            {
                Ok(())
            } else {
                Err(crate::Error::FailedToSetProperty(
                    PropertySetAttempt::BuilderConfigNbComputeCapabilities,
                ))
            }
        } else {
            Ok(())
        }
    }

    /// See [IBuilderConfig::getNbComputeCapabilities]
    pub fn get_nb_compute_capabilities(&self) -> i32 {
        if cfg!(not(feature = "mock")) {
            self.inner.getNbComputeCapabilities()
        } else {
            0
        }
    }

    /// See [IBuilderConfig::setComputeCapability]
    pub fn set_compute_capability(
        &mut self,
        compute_capability: ComputeCapability,
        index: i32,
    ) -> crate::Result<()> {
        if cfg!(not(feature = "mock")) {
            if self
                .inner
                .pin_mut()
                .setComputeCapability(compute_capability.into(), index)
            {
                Ok(())
            } else {
                Err(crate::Error::FailedToSetProperty(
                    PropertySetAttempt::BuilderConfigComputeCapability,
                ))
            }
        } else {
            Ok(())
        }
    }

    /// See [IBuilderConfig::getComputeCapability]
    pub fn get_compute_capability(&self, index: i32) -> ComputeCapability {
        if cfg!(not(feature = "mock")) {
            self.inner.getComputeCapability(index).into()
        } else {
            ComputeCapability::kNONE
        }
    }
}

#[cfg(test)]
#[cfg(not(feature = "mock"))]
mod tests {
    use crate::builder::MemoryPoolType;
    use crate::interfaces::{HandleProgress, ProgressMonitor};
    use crate::{Builder, DataType, Logger, NetworkDefinition};
    use std::ops::ControlFlow;
    use std::sync::atomic::{AtomicU32, Ordering};

    const NUM_LAYERS: usize = 40;

    /// Progress monitor that writes to stdout and cancels the build after a few steps.
    struct StdoutProgressMonitor {
        step_count: AtomicU32,
        cancel_after: u32,
    }

    impl StdoutProgressMonitor {
        fn new(cancel_after: u32) -> Self {
            Self {
                step_count: AtomicU32::new(0),
                cancel_after,
            }
        }
    }

    impl HandleProgress for StdoutProgressMonitor {
        fn phase_start(&self, phase_name: &str, parent_phase: Option<&str>, num_steps: i32) {
            println!(
                "[progress] phase_start phase={:?} parent={:?} num_steps={}",
                phase_name, parent_phase, num_steps
            );
        }

        fn step_complete(&self, phase_name: &str, step: i32) -> ControlFlow<()> {
            let n = self.step_count.fetch_add(1, Ordering::SeqCst);
            println!(
                "[progress] step_complete phase={:?} step={}",
                phase_name, step
            );
            if n + 1 >= self.cancel_after {
                println!("[progress] cancel requested after {} steps", n + 1);
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }

        fn phase_finish(&self, phase_name: &str) {
            println!("[progress] phase_finish phase={:?}", phase_name);
        }
    }

    /// Build a network with many repeated identity layers, each named.
    fn build_heavy_network(logger: &Logger) -> crate::Result<(Builder<'_>, NetworkDefinition<'_>)> {
        let mut builder = Builder::new(logger)?;
        let mut network = builder.create_network(0)?;

        let mut tensor = network.add_input("input", DataType::kFLOAT, &[1, 4])?;
        for i in 0..NUM_LAYERS {
            let mut layer = network.add_identity(&mut tensor)?;
            layer.set_name(&mut network, &format!("layer_{}", i))?;
            tensor = layer.get_output(&network, 0)?;
        }
        tensor.set_name(&mut network, "output")?;
        network.mark_output(&mut tensor);

        Ok((builder, network))
    }

    #[test]
    fn set_progress_monitor_cancel_build() {
        let logger = Logger::stderr().expect("logger");
        let (mut builder, mut network) = build_heavy_network(&logger).expect("build network");

        let mut config = builder.create_config().expect("config");
        config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, 1 << 24);

        let monitor = StdoutProgressMonitor::new(3);
        config.set_progress_monitor(ProgressMonitor::new(Box::new(monitor)).unwrap());

        let result = builder.build_serialized_network(&mut network, &mut config);

        assert!(
            result.is_err(),
            "build should fail (cancelled by progress monitor)"
        );
    }

    #[test]
    fn set_progress_monitor_progress_to_stdout() {
        let logger = Logger::stderr().expect("logger");
        let (mut builder, mut network) = build_heavy_network(&logger).expect("build network");

        let mut config = builder.create_config().expect("config");
        config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, 1 << 24);

        let monitor = StdoutProgressMonitor::new(10000);
        config.set_progress_monitor(ProgressMonitor::new(Box::new(monitor)).unwrap());

        let result = builder.build_serialized_network(&mut network, &mut config);

        assert!(result.is_ok(), "build should succeed when not cancelling");
    }
}
