//! CUDA memory management utilities using cudarc

use crate::error::{Error, Result};

#[cfg(all(not(feature = "mock"), feature = "use-cudarc"))]
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

/// RAII wrapper for CUDA device memory
pub struct DeviceBuffer {
    #[cfg(all(not(feature = "mock"), feature = "use-cudarc"))]
    ptr: CudaSlice<u8>,
    #[cfg(all(not(feature = "mock"), feature = "use-cudarc"))]
    device: std::sync::Arc<CudaDevice>,
    #[cfg(feature = "mock")]
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

        #[cfg(all(not(feature = "mock"), feature = "use-cudarc"))]
        {
            // Get CUDA device (device 0 by default)
            let device = CudaDevice::new(0)
                .map_err(|e| Error::Cuda(format!("Failed to initialize CUDA device: {:?}", e)))?;

            // Allocate device memory
            let ptr = device
                .alloc_zeros::<u8>(size)
                .map_err(|e| Error::Cuda(format!("Failed to allocate CUDA memory: {:?}", e)))?;

            Ok(DeviceBuffer { ptr, device, size })
        }
    }

    /// Get the raw device pointer (for TensorRT interop)
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        #[cfg(feature = "mock")]
        {
            self.ptr
        }

        #[cfg(all(not(feature = "mock"), feature = "use-cudarc"))]
        {
            *self.ptr.device_ptr() as *mut std::ffi::c_void
        }
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

        #[cfg(all(not(feature = "mock"), feature = "use-cudarc"))]
        {
            // Use cudarc's safe copy
            self.device
                .htod_copy_into(data.to_vec(), &mut self.ptr)
                .map_err(|e| Error::Cuda(format!("Failed to copy to device: {:?}", e)))
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

        #[cfg(all(not(feature = "mock"), feature = "use-cudarc"))]
        {
            // Use cudarc's safe copy
            self.device
                .dtoh_sync_copy_into(&self.ptr, data)
                .map_err(|e| Error::Cuda(format!("Failed to copy from device: {:?}", e)))
        }
    }
}

// DeviceBuffer automatically frees memory when dropped (cudarc handles this)
impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "mock")]
        {
            if !self.ptr.is_null() {
                let mut error_msg = [0i8; 1024];
                unsafe {
                    let _ =
                        trtx_sys::trtx_cuda_free(self.ptr, error_msg.as_mut_ptr(), error_msg.len());
                }
            }
        }

        // In non-mock mode, cudarc's CudaSlice automatically frees on drop
        #[cfg(all(not(feature = "mock"), feature = "use-cudarc"))]
        {
            // Nothing to do - cudarc handles cleanup
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

    #[cfg(all(not(feature = "mock"), feature = "use-cudarc"))]
    {
        // Get device and synchronize
        let device = CudaDevice::new(0)
            .map_err(|e| Error::Cuda(format!("Failed to get CUDA device: {:?}", e)))?;

        device
            .synchronize()
            .map_err(|e| Error::Cuda(format!("Failed to synchronize device: {:?}", e)))
    }
}

/// Get the default CUDA stream
pub fn get_default_stream() -> *mut std::ffi::c_void {
    #[cfg(feature = "mock")]
    {
        unsafe { trtx_sys::trtx_cuda_get_default_stream() }
    }

    #[cfg(all(not(feature = "mock"), feature = "use-cudarc"))]
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
