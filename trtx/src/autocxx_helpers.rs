//! Helper utilities for working with autocxx-generated TensorRT bindings
//!
//! autocxx requires using `Pin<&mut T>` to call C++ methods on objects.
//! These helpers reduce boilerplate and make the code cleaner.

use std::pin::Pin;

/// Convert a raw pointer to a pinned mutable reference
///
/// # Safety
///
/// The caller must ensure:
/// - `ptr` is not null
/// - `ptr` points to a valid, initialized object
/// - The object remains valid for the lifetime of the returned Pin
/// - No other code mutates the object through another reference
///
/// # Example
///
/// ```ignore
/// unsafe {
///     let builder_ptr: *mut std::ffi::c_void = /* from C++ */;
///     let network = pin_mut(builder_ptr as *mut trtx_sys::nvinfer1::IBuilder)
///         .createNetworkV2(flags);
/// }
/// ```
#[inline]
pub unsafe fn pin_mut<T>(ptr: *mut T) -> Pin<&'static mut T> {
    Pin::new_unchecked(&mut *ptr)
}

/// Cast a void pointer and pin it in one operation
///
/// # Safety
///
/// Same safety requirements as `pin_mut`, plus:
/// - The cast from `*mut c_void` to `*mut T` must be valid
///
/// # Example
///
/// ```ignore
/// unsafe {
///     let void_ptr: *mut std::ffi::c_void = /* from C++ */;
///     let network = cast_and_pin::<trtx_sys::nvinfer1::IBuilder>(void_ptr)
///         .createNetworkV2(flags);
/// }
/// ```
#[inline]
pub unsafe fn cast_and_pin<T>(void_ptr: *mut std::ffi::c_void) -> Pin<&'static mut T> {
    pin_mut(void_ptr as *mut T)
}

/// Helper macro to reduce boilerplate for autocxx method calls
///
/// # Example
///
/// ```ignore
/// // Instead of:
/// unsafe {
///     let builder_ref = &mut *(builder_ptr as *mut trtx_sys::nvinfer1::IBuilder);
///     let mut pinned = Pin::new_unchecked(builder_ref);
///     pinned.as_mut().createNetworkV2(flags)
/// }
///
/// // Write:
/// autocxx_call!(builder_ptr, nvinfer1::IBuilder, createNetworkV2(flags))
/// ```
#[macro_export]
macro_rules! autocxx_call {
    ($ptr:expr, $type:path, $method:ident($($args:expr),*)) => {
        unsafe {
            $crate::autocxx_helpers::cast_and_pin::<trtx_sys::$type>($ptr)
                .$method($($args),*)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_mut_compiles() {
        // This test just verifies the function signature compiles
        // We can't actually test it without a real C++ object

        let _f: unsafe fn(*mut u32) -> Pin<&'static mut u32> = pin_mut;
    }

    #[test]
    fn test_cast_and_pin_compiles() {
        let _f: unsafe fn(*mut std::ffi::c_void) -> Pin<&'static mut u32> = cast_and_pin;
    }
}
