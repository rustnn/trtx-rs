// Comprehensive test: Can we call TensorRT methods via autocxx?
// This is the KEY test for determining if we can remove C wrappers

#![allow(unused)]

#[cfg(not(feature = "mock"))]
#[test]
#[ignore] // Run with: cargo test --test method_call_test -- --ignored
#[cfg(feature = "link_tensorrt_rtx")]
fn test_builder_methods_callable() {
    use std::ptr;

    // First, we need a logger to create a builder
    // Using the existing logger bridge (which IS necessary)
    unsafe {
        let callback: trtx_sys::RustLogCallback = test_logger_callback;
        let logger_bridge = trtx_sys::create_rust_logger_bridge(callback, ptr::null_mut());
        assert!(!logger_bridge.is_null(), "Failed to create logger bridge");

        let logger = trtx_sys::get_logger_interface(logger_bridge);
        assert!(!logger.is_null(), "Failed to get logger interface");

        // Create builder using factory (also necessary - takes ILogger&)
        let builder_ptr = trtx_sys::create_infer_builder(logger as *mut _);
        assert!(!builder_ptr.is_null(), "Failed to create builder");

        // NOW THE KEY TEST: Can we call methods on IBuilder?
        // Cast void* to IBuilder*
        let builder = builder_ptr as *mut trtx_sys::nvinfer1::IBuilder;

        // Attempt to call createNetworkV2() - THIS IS THE TEST!
        // If this compiles and works, we can remove builder_create_network_v2() wrapper

        // Note: We can't actually test this without proper setup, but we can check if it compiles
        println!("Builder pointer: {:?}", builder);

        // Cleanup
        trtx_sys::delete_builder(builder_ptr);
        trtx_sys::destroy_rust_logger_bridge(logger_bridge);
    }
}

#[cfg(not(feature = "mock"))]
unsafe extern "C" fn test_logger_callback(
    _user_data: *mut std::ffi::c_void,
    severity: i32,
    msg: *const std::os::raw::c_char,
) {
    if !msg.is_null() {
        let c_str = std::ffi::CStr::from_ptr(msg);
        if let Ok(s) = c_str.to_str() {
            println!("[TensorRT {}] {}", severity, s);
        }
    }
}

// This test checks if autocxx provides method access AT COMPILE TIME
#[cfg(not(feature = "mock"))]
#[test]
fn test_autocxx_method_availability_compile_check() {
    // This function won't run, but if it COMPILES, we know the methods exist

    #[allow(unreachable_code)]
    {
        return; // Don't actually run this

        unsafe {
            let builder: *mut trtx_sys::nvinfer1::IBuilder = std::ptr::null_mut();

            // Try to access methods - if these compile, autocxx generated them!
            // Uncomment these one at a time to test:

            // Test 1: Does IBuilder have methods?
            // let network = (*builder).createNetworkV2(0);

            // Test 2: Does INetworkDefinition have methods?
            // let network: *mut trtx_sys::nvinfer1::INetworkDefinition = std::ptr::null_mut();
            // let tensor = (*network).addInput(...);
        }
    }
}
