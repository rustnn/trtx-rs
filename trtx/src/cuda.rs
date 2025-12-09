//! CUDA memory management utilities

use crate::error::{Error, Result};
use trtx_sys::*;

/// RAII wrapper for CUDA device memory
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl DeviceBuffer {
    /// Allocate CUDA device memory
    pub fn new(size: usize) -> Result<Self> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut error_msg = [0i8; 1024];

        let result =
            unsafe { trtx_cuda_malloc(&mut ptr, size, error_msg.as_mut_ptr(), error_msg.len()) };

        if result != TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        Ok(DeviceBuffer { ptr, size })
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

        let mut error_msg = [0i8; 1024];

        let result = unsafe {
            trtx_cuda_memcpy_host_to_device(
                self.ptr,
                data.as_ptr() as *const std::ffi::c_void,
                data.len(),
                error_msg.as_mut_ptr(),
                error_msg.len(),
            )
        };

        if result != TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(Error::InvalidArgument(
                "Data size exceeds buffer size".to_string(),
            ));
        }

        let mut error_msg = [0i8; 1024];

        let result = unsafe {
            trtx_cuda_memcpy_device_to_host(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr,
                data.len(),
                error_msg.as_mut_ptr(),
                error_msg.len(),
            )
        };

        if result != TRTX_SUCCESS as i32 {
            return Err(Error::from_ffi(result, &error_msg));
        }

        Ok(())
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let mut error_msg = [0i8; 1024];
            unsafe {
                let _ = trtx_cuda_free(self.ptr, error_msg.as_mut_ptr(), error_msg.len());
            }
        }
    }
}

unsafe impl Send for DeviceBuffer {}

/// Synchronize CUDA device
pub fn synchronize() -> Result<()> {
    let mut error_msg = [0i8; 1024];

    let result = unsafe { trtx_cuda_synchronize(error_msg.as_mut_ptr(), error_msg.len()) };

    if result != TRTX_SUCCESS as i32 {
        return Err(Error::from_ffi(result, &error_msg));
    }

    Ok(())
}

/// Get the default CUDA stream
pub fn get_default_stream() -> *mut std::ffi::c_void {
    unsafe { trtx_cuda_get_default_stream() }
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
