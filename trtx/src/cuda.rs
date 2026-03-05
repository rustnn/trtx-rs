//! CUDA memory management utilities

use crate::error::{Error, Result};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

/// RAII wrapper for CUDA device memory
pub struct DeviceBuffer {
    ptr: CudaSlice<u8>,
    device: std::sync::Arc<CudaDevice>,
    size: usize,
}

impl DeviceBuffer {
    pub fn new(size: usize) -> Result<Self> {
        let device = CudaDevice::new(0)
            .map_err(|e| Error::Cuda(format!("Failed to initialize CUDA device: {:?}", e)))?;
        let ptr = device
            .alloc_zeros::<u8>(size)
            .map_err(|e| Error::Cuda(format!("Failed to allocate CUDA memory: {:?}", e)))?;
        Ok(DeviceBuffer { ptr, device, size })
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        *self.ptr.device_ptr() as *mut std::ffi::c_void
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
        self.device
            .htod_copy_into(data.to_vec(), &mut self.ptr)
            .map_err(|e| Error::Cuda(format!("Failed to copy to device: {:?}", e)))
    }

    pub fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(Error::InvalidArgument(
                "Data size exceeds buffer size".to_string(),
            ));
        }
        self.device
            .dtoh_sync_copy_into(&self.ptr, data)
            .map_err(|e| Error::Cuda(format!("Failed to copy from device: {:?}", e)))
    }
}

unsafe impl Send for DeviceBuffer {}

/// Synchronize CUDA device
pub fn synchronize() -> Result<()> {
    let device = CudaDevice::new(0)
        .map_err(|e| Error::Cuda(format!("Failed to get CUDA device: {:?}", e)))?;
    device
        .synchronize()
        .map_err(|e| Error::Cuda(format!("Failed to synchronize device: {:?}", e)))
}

/// Get the default CUDA stream
pub fn get_default_stream() -> *mut std::ffi::c_void {
    std::ptr::null_mut()
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
