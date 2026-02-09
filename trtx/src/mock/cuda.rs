//! Mock CUDA implementations

use crate::error::{Error, Result};

use super::from_ffi;

/// RAII wrapper for CUDA device memory (mock mode)
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl DeviceBuffer {
    pub fn new(size: usize) -> Result<Self> {
        let ptr = cuda_malloc(size)?;
        Ok(DeviceBuffer { ptr, size })
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(Error::InvalidArgument(
                "Data size exceeds buffer size".to_string(),
            ));
        }
        memcpy_host_to_device(self.ptr, data)
    }

    pub fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(Error::InvalidArgument(
                "Data size exceeds buffer size".to_string(),
            ));
        }
        memcpy_device_to_host(self.ptr, data)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        cuda_free(self.ptr);
    }
}

unsafe impl Send for DeviceBuffer {}

/// Synchronize CUDA device
pub fn synchronize() -> Result<()> {
    let mut error_msg = [0i8; 1024];
    let result =
        unsafe { trtx_sys::trtx_cuda_synchronize(error_msg.as_mut_ptr(), error_msg.len()) };
    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }
    Ok(())
}

/// Get default CUDA stream
pub fn get_default_stream() -> *mut std::ffi::c_void {
    unsafe { trtx_sys::trtx_cuda_get_default_stream() }
}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

fn cuda_malloc(size: usize) -> Result<*mut std::ffi::c_void> {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut error_msg = [0i8; 1024];
    let result = unsafe {
        trtx_sys::trtx_cuda_malloc(&mut ptr, size, error_msg.as_mut_ptr(), error_msg.len())
    };
    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }
    Ok(ptr)
}

fn memcpy_host_to_device(ptr: *mut std::ffi::c_void, data: &[u8]) -> Result<()> {
    let mut error_msg = [0i8; 1024];
    let result = unsafe {
        trtx_sys::trtx_cuda_memcpy_host_to_device(
            ptr,
            data.as_ptr() as *const std::ffi::c_void,
            data.len(),
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };
    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }
    Ok(())
}

fn memcpy_device_to_host(ptr: *mut std::ffi::c_void, data: &mut [u8]) -> Result<()> {
    let mut error_msg = [0i8; 1024];
    let result = unsafe {
        trtx_sys::trtx_cuda_memcpy_device_to_host(
            data.as_mut_ptr() as *mut std::ffi::c_void,
            ptr,
            data.len(),
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };
    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }
    Ok(())
}

fn cuda_free(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        let mut error_msg = [0i8; 1024];
        unsafe {
            let _ = trtx_sys::trtx_cuda_free(ptr, error_msg.as_mut_ptr(), error_msg.len());
        }
    }
}
