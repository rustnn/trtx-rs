//! CUDA memory management utilities

use crate::error::{Error, Result};

/// Helper function to convert CUDA error code to error string
#[cfg(not(feature = "mock"))]
fn cuda_error_to_string(error_code: i32) -> String {
    unsafe {
        let c_str = trtx_sys::cuda_get_error_string_wrapper(error_code);
        if c_str.is_null() {
            format!("CUDA error code: {}", error_code)
        } else {
            std::ffi::CStr::from_ptr(c_str)
                .to_str()
                .unwrap_or("Unknown CUDA error")
                .to_string()
        }
    }
}

/// RAII wrapper for CUDA device memory
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl DeviceBuffer {
    /// Allocate CUDA device memory
    pub fn new(size: usize) -> Result<Self> {
        #[cfg(feature = "mock")]
        {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_cuda_malloc(&mut ptr, size, error_msg.as_mut_ptr(), error_msg.len())
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(DeviceBuffer { ptr, size })
        }

        #[cfg(not(feature = "mock"))]
        {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();

            let result = unsafe { trtx_sys::cuda_malloc_wrapper(&mut ptr, size) };

            if result != trtx_sys::CUDA_SUCCESS {
                return Err(Error::Cuda(cuda_error_to_string(result)));
            }

            Ok(DeviceBuffer { ptr, size })
        }
    }

    /// Get the raw device pointer
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(Error::InvalidArgument(
                "Data size exceeds buffer size".to_string(),
            ));
        }

        #[cfg(feature = "mock")]
        {
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_cuda_memcpy_host_to_device(
                    self.ptr,
                    data.as_ptr() as *const std::ffi::c_void,
                    data.len(),
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(())
        }

        #[cfg(not(feature = "mock"))]
        {
            let result = unsafe {
                trtx_sys::cuda_memcpy_wrapper(
                    self.ptr,
                    data.as_ptr() as *const std::ffi::c_void,
                    data.len(),
                    trtx_sys::CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            };

            if result != trtx_sys::CUDA_SUCCESS {
                return Err(Error::Cuda(cuda_error_to_string(result)));
            }

            Ok(())
        }
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(Error::InvalidArgument(
                "Data size exceeds buffer size".to_string(),
            ));
        }

        #[cfg(feature = "mock")]
        {
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_cuda_memcpy_device_to_host(
                    data.as_mut_ptr() as *mut std::ffi::c_void,
                    self.ptr,
                    data.len(),
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(())
        }

        #[cfg(not(feature = "mock"))]
        {
            let result = unsafe {
                trtx_sys::cuda_memcpy_wrapper(
                    data.as_mut_ptr() as *mut std::ffi::c_void,
                    self.ptr,
                    data.len(),
                    trtx_sys::CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            };

            if result != trtx_sys::CUDA_SUCCESS {
                return Err(Error::Cuda(cuda_error_to_string(result)));
            }

            Ok(())
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            #[cfg(feature = "mock")]
            {
                let mut error_msg = [0i8; 1024];
                unsafe {
                    let _ =
                        trtx_sys::trtx_cuda_free(self.ptr, error_msg.as_mut_ptr(), error_msg.len());
                }
            }

            #[cfg(not(feature = "mock"))]
            {
                unsafe {
                    let _ = trtx_sys::cuda_free_wrapper(self.ptr);
                }
            }
        }
    }
}

unsafe impl Send for DeviceBuffer {}

/// Synchronize CUDA device
pub fn synchronize() -> Result<()> {
    #[cfg(feature = "mock")]
    {
        let mut error_msg = [0i8; 1024];

        let result =
            unsafe { trtx_sys::trtx_cuda_synchronize(error_msg.as_mut_ptr(), error_msg.len()) };

        if result != trtx_sys::TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        Ok(())
    }

    #[cfg(not(feature = "mock"))]
    {
        let result = unsafe { trtx_sys::cuda_device_synchronize_wrapper() };

        if result != trtx_sys::CUDA_SUCCESS {
            return Err(Error::Cuda(cuda_error_to_string(result)));
        }

        Ok(())
    }
}

/// Get the default CUDA stream
pub fn get_default_stream() -> *mut std::ffi::c_void {
    #[cfg(feature = "mock")]
    {
        unsafe { trtx_sys::trtx_cuda_get_default_stream() }
    }

    #[cfg(not(feature = "mock"))]
    {
        // Default stream is nullptr in CUDA
        std::ptr::null_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_buffer_allocation() {
        let buffer = DeviceBuffer::new(1024);
        assert!(buffer.is_ok());

        let buffer = buffer.unwrap();
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_device_buffer_copy() {
        let mut buffer = DeviceBuffer::new(256).unwrap();

        let host_data = vec![42u8; 256];
        assert!(buffer.copy_from_host(&host_data).is_ok());

        let mut output = vec![0u8; 256];
        assert!(buffer.copy_to_host(&mut output).is_ok());

        assert_eq!(host_data, output);
    }

    #[test]
    fn test_synchronize() {
        assert!(synchronize().is_ok());
    }
}
