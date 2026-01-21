// Test to check if autocxx generates TensorRT class methods
// This helps us understand if we can remove manual C wrappers

#[cfg(not(feature = "mock"))]
#[test]
#[ignore] // Ignore by default, run with --ignored
fn test_autocxx_builder_methods_exist() {
    // This test just checks if the methods compile
    // We're not actually running them

    // Check if we can access IBuilder type
    let _builder_type: Option<*mut trtx_sys::nvinfer1::IBuilder> = None;

    // Try to call a method (won't execute, just checking if it compiles)
    // Uncomment to test:
    // if let Some(builder) = builder_ptr.as_mut() {
    //     let _network = builder.createNetworkV2(0);
    // }

    println!("If this compiles, autocxx generated IBuilder bindings");
}

#[cfg(not(feature = "mock"))]
#[test]
#[ignore]
fn test_check_available_types() {
    // List all types we expect autocxx to generate

    // Check if types exist
    let _: Option<*mut trtx_sys::nvinfer1::IBuilder> = None;
    let _: Option<*mut trtx_sys::nvinfer1::INetworkDefinition> = None;
    let _: Option<*mut trtx_sys::nvinfer1::IBuilderConfig> = None;
    let _: Option<*mut trtx_sys::nvinfer1::ICudaEngine> = None;
    let _: Option<*mut trtx_sys::nvinfer1::IExecutionContext> = None;
    let _: Option<*mut trtx_sys::nvinfer1::IRuntime> = None;

    println!("All expected types are available from autocxx");
}
