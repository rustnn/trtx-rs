//! Real CUDA implementation (uses cudarc when use-cudarc feature is enabled)

use crate::error::{Error, Result};

#[cfg(feature = "use-cudarc")]
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

/// RAII wrapper for CUDA device memory (real mode)
pub struct DeviceBuffer {
    #[cfg(feature = "use-cudarc")]
    ptr: CudaSlice<u8>,
    #[cfg(feature = "use-cudarc")]
    device: std::sync::Arc<CudaDevice>,
    size: usize,
}

impl DeviceBuffer {
    pub fn new(size: usize) -> Result<Self> {
        #[cfg(feature = "use-cudarc")]
        {
            let device = CudaDevice::new(0)
                .map_err(|e| Error::Cuda(format!("Failed to initialize CUDA device: {:?}", e)))?;
            let ptr = device
                .alloc_zeros::<u8>(size)
                .map_err(|e| Error::Cuda(format!("Failed to allocate CUDA memory: {:?}", e)))?;
            Ok(DeviceBuffer { ptr, device, size })
        }

        #[cfg(not(feature = "use-cudarc"))]
        {
            let _ = size;
            Err(Error::Cuda(
                "Real mode requires use-cudarc feature for CUDA operations".to_string(),
            ))
        }
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        #[cfg(feature = "use-cudarc")]
        {
            *self.ptr.device_ptr() as *mut std::ffi::c_void
        }
        #[cfg(not(feature = "use-cudarc"))]
        {
            std::ptr::null_mut()
        }
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
        #[cfg(feature = "use-cudarc")]
        {
            self.device
                .htod_copy_into(data.to_vec(), &mut self.ptr)
                .map_err(|e| Error::Cuda(format!("Failed to copy to device: {:?}", e)))
        }
        #[cfg(not(feature = "use-cudarc"))]
        {
            let _ = data;
            Err(Error::Cuda("use-cudarc required".to_string()))
        }
    }

    pub fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(Error::InvalidArgument(
                "Data size exceeds buffer size".to_string(),
            ));
        }
        #[cfg(feature = "use-cudarc")]
        {
            self.device
                .dtoh_sync_copy_into(&self.ptr, data)
                .map_err(|e| Error::Cuda(format!("Failed to copy from device: {:?}", e)))
        }
        #[cfg(not(feature = "use-cudarc"))]
        {
            let _ = data;
            Err(Error::Cuda("use-cudarc required".to_string()))
        }
    }
}

unsafe impl Send for DeviceBuffer {}

/// Synchronize CUDA device
pub fn synchronize() -> Result<()> {
    #[cfg(feature = "use-cudarc")]
    {
        let device = CudaDevice::new(0)
            .map_err(|e| Error::Cuda(format!("Failed to get CUDA device: {:?}", e)))?;
        device
            .synchronize()
            .map_err(|e| Error::Cuda(format!("Failed to synchronize device: {:?}", e)))
    }
    #[cfg(not(feature = "use-cudarc"))]
    {
        Err(Error::Cuda("use-cudarc required".to_string()))
    }
}

/// Get the default CUDA stream
pub fn get_default_stream() -> *mut std::ffi::c_void {
    std::ptr::null_mut()
}
