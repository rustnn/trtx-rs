// Ultra-simple test: What does autocxx actually generate for IBuilder?

/// Helper to pin a raw pointer (same as in trtx crate)
#[cfg(not(feature = "mock"))]
unsafe fn pin_mut<T>(ptr: *mut T) -> std::pin::Pin<&'static mut T> {
    std::pin::Pin::new_unchecked(&mut *ptr)
}

#[cfg(not(feature = "mock"))]
#[test]
fn check_what_autocxx_provides() {
    // Just try to reference methods and see what the compiler tells us

    unsafe {
        // Get a null pointer of the right type
        let builder: *mut trtx_sys::nvinfer1::IBuilder = std::ptr::null_mut();

        // Can we dereference it to access methods?
        if !builder.is_null() {
            // Using our helper function (same pattern as trtx crate)
            let _network = pin_mut(builder).createNetworkV2(0);

            println!("SUCCESS! Called createNetworkV2 via autocxx!");
        }

        println!("Test compiled - Pin helper works!");
    }
}
