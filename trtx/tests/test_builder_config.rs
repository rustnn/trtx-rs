// Simple test to verify BuilderConfig methods are accessible
#[cfg(test)]
mod tests {
    use trtx::builder::{
        Builder, BuilderFlag, DeviceType, EngineCapability, HardwareCompatibilityLevel,
        MemoryPoolType, PreviewFeature, ProfilingVerbosity, RuntimePlatform,
        TilingOptimizationLevel,
    };

    use trtx::logger::Logger;
    #[cfg(not(feature = "enterprise"))]
    use trtx::ComputeCapability;

    #[test]
    fn test_builder_config_methods() {
        #[cfg(feature = "dlopen_tensorrt_rtx")]
        trtx::dynamically_load_tensorrt(None::<String>).unwrap();

        let logger = Logger::stderr().unwrap();
        let mut builder = Builder::new(&logger).unwrap();
        let mut config = builder.create_config().unwrap();

        // Test timing iterations
        config.set_avg_timing_iterations(5);
        #[cfg(not(feature = "mock"))]
        assert_eq!(config.get_avg_timing_iterations(), 5);

        // Test engine capability
        config.set_engine_capability(EngineCapability::kSTANDARD);
        #[cfg(not(feature = "mock"))]
        assert_eq!(config.get_engine_capability(), EngineCapability::kSTANDARD);

        // Test flags
        config.set_flags(0);
        #[cfg(not(feature = "mock"))]
        assert_eq!(config.get_flags(), 0);
        config.set_flag(BuilderFlag::kDEBUG);
        #[cfg(not(feature = "mock"))]
        assert!(config.get_flag(BuilderFlag::kDEBUG));
        config.clear_flag(BuilderFlag::kDEBUG);
        #[cfg(not(feature = "mock"))]
        assert!(!config.get_flag(BuilderFlag::kDEBUG));

        // Test DLA core
        config.set_dla_core(-1);
        #[cfg(not(feature = "mock"))]
        assert_eq!(config.get_dla_core(), -1);

        // Test device type
        config.set_default_device_type(DeviceType::kGPU);
        #[cfg(not(feature = "mock"))]
        assert_eq!(config.get_default_device_type(), DeviceType::kGPU);

        // Test optimization profiles
        assert_eq!(config.get_nb_optimization_profiles(), 0);

        // Test tactic sources (kEDGE_MASK_CONVOLUTIONS = 3)
        let sources = 1u32 << 3;
        config.set_tactic_sources(sources).unwrap();
        #[cfg(not(feature = "mock"))]
        assert_eq!(config.get_tactic_sources(), sources);

        config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, 0);
        let limit = config.get_memory_pool_limit(MemoryPoolType::kWORKSPACE);
        assert_eq!(limit, 0); // Default in mock

        // Test preview feature
        config.set_preview_feature(PreviewFeature::kALIASED_PLUGIN_IO_10_03, true);
        #[cfg(not(feature = "mock"))]
        assert!(config.get_preview_feature(PreviewFeature::kALIASED_PLUGIN_IO_10_03));

        // Test builder optimization level
        config.set_builder_optimization_level(5);
        #[cfg(not(feature = "mock"))]
        assert_eq!(config.get_builder_optimization_level(), 5);

        // Test hardware compatibility level
        config.set_hardware_compatibility_level(HardwareCompatibilityLevel::kNONE);
        #[cfg(not(feature = "mock"))]
        assert_eq!(
            config.get_hardware_compatibility_level(),
            HardwareCompatibilityLevel::kNONE
        );

        // Test max aux streams
        config.set_max_aux_streams(2);
        #[cfg(not(feature = "mock"))]
        assert_eq!(config.get_max_aux_streams(), 2);

        // Test runtime platform
        config.set_runtime_platform(RuntimePlatform::kSAME_AS_BUILD);
        #[cfg(not(feature = "mock"))]
        assert_eq!(
            config.get_runtime_platform(),
            RuntimePlatform::kSAME_AS_BUILD
        );

        // Test max nb tactics
        config.set_max_nb_tactics(10);
        #[cfg(not(feature = "mock"))]
        assert_eq!(config.get_max_nb_tactics(), 10);

        // Test tiling optimization level
        config
            .set_tiling_optimization_level(TilingOptimizationLevel::kFAST)
            .unwrap();
        #[cfg(not(feature = "mock"))]
        assert_eq!(
            config.get_tiling_optimization_level(),
            TilingOptimizationLevel::kFAST
        );

        // Test L2 limit for tiling
        config.set_l2_limit_for_tiling(1024).unwrap();
        #[cfg(not(feature = "mock"))]
        assert_eq!(config.get_l2_limit_for_tiling(), 1024);

        // Test compute capabilities
        #[cfg(not(feature = "enterprise"))]
        {
            config.set_nb_compute_capabilities(1).unwrap();
            #[cfg(not(feature = "mock"))]
            assert_eq!(config.get_nb_compute_capabilities(), 1);
            config
                .set_compute_capability(ComputeCapability::kCURRENT, 0)
                .unwrap();
            #[cfg(not(feature = "mock"))]
            assert_eq!(
                config.get_compute_capability(0),
                ComputeCapability::kCURRENT
            );
        }

        // Test profiling verbosity
        config.set_profiling_verbosity(ProfilingVerbosity::kDETAILED);
        #[cfg(not(feature = "mock"))]
        assert_eq!(
            config.get_profiling_verbosity(),
            ProfilingVerbosity::kDETAILED
        );

        // Test reset
        config.reset();

        println!("All BuilderConfig methods working correctly!");
    }
}
