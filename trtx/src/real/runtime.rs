//! Real TensorRT runtime implementation

use crate::error::{Error, Result};
use crate::logger::Logger;
use std::ffi::CStr;

/// CUDA engine (real mode)
pub struct CudaEngine {
    inner: *mut std::ffi::c_void,
}

impl CudaEngine {
    pub fn get_nb_io_tensors(&self) -> Result<i32> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid engine".to_string()));
        }
        let count = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ICudaEngine>(self.inner)
                .getNbIOTensors()
        };
        Ok(count)
    }

    pub fn get_tensor_name(&self, index: i32) -> Result<String> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid engine".to_string()));
        }
        let name_ptr = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ICudaEngine>(self.inner)
                .getIOTensorName(index)
        };
        if name_ptr.is_null() {
            return Err(Error::InvalidArgument("Invalid tensor index".to_string()));
        }
        Ok(unsafe { CStr::from_ptr(name_ptr) }.to_str()?.to_string())
    }

    pub fn get_tensor_shape(&self, name: &str) -> Result<Vec<i64>> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid engine".to_string()));
        }
        let name_cstr = std::ffi::CString::new(name)?;
        let dims = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ICudaEngine>(self.inner)
                .getTensorShape(name_cstr.as_ptr())
        };
        let nb_dims = dims.nbDims as usize;
        if nb_dims > 8 {
            return Err(Error::Runtime("Tensor has too many dimensions".to_string()));
        }
        let mut shape = Vec::with_capacity(nb_dims);
        for i in 0..nb_dims {
            shape.push(dims.d[i]);
        }
        Ok(shape)
    }

    pub fn create_execution_context(&self) -> Result<ExecutionContext<'_>> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid engine".to_string()));
        }
        let context_ptr = unsafe {
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ICudaEngine>(self.inner)
                .createExecutionContext(
                    trtx_sys::nvinfer1::ExecutionContextAllocationStrategy::kSTATIC,
                )
        };
        if context_ptr.is_null() {
            return Err(Error::Runtime(
                "Failed to create execution context".to_string(),
            ));
        }
        Ok(ExecutionContext {
            inner: context_ptr as *mut _,
            _engine: std::marker::PhantomData,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn as_ptr(&self) -> *const std::ffi::c_void {
        self.inner
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_sys::delete_engine(self.inner);
            }
        }
    }
}

unsafe impl Send for CudaEngine {}
unsafe impl Sync for CudaEngine {}

/// Execution context (real mode)
pub struct ExecutionContext<'a> {
    inner: *mut std::ffi::c_void,
    _engine: std::marker::PhantomData<&'a CudaEngine>,
}

impl<'a> ExecutionContext<'a> {
    /// Binds a tensor to a device memory address.
    ///
    /// # Safety
    /// `data` must point to valid CUDA memory with at least the tensor's size in bytes,
    /// and remain valid for the duration of inference.
    pub unsafe fn set_tensor_address(
        &mut self,
        name: &str,
        data: *mut std::ffi::c_void,
    ) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid execution context".to_string()));
        }
        let name_cstr = std::ffi::CString::new(name)?;
        let success =
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::IExecutionContext>(
                self.inner,
            )
            .setTensorAddress(name_cstr.as_ptr(), data as *mut _);
        if !success {
            return Err(Error::Runtime("Failed to set tensor address".to_string()));
        }
        Ok(())
    }

    /// Enqueues inference on the given CUDA stream.
    ///
    /// # Safety
    /// `cuda_stream` must be a valid CUDA stream, and all tensor addresses must
    /// point to valid device memory.
    pub unsafe fn enqueue_v3(&mut self, cuda_stream: *mut std::ffi::c_void) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid execution context".to_string()));
        }
        let success =
            crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::IExecutionContext>(
                self.inner,
            )
            .enqueueV3(cuda_stream as *mut _);
        if !success {
            return Err(Error::Runtime("Failed to enqueue inference".to_string()));
        }
        Ok(())
    }
}

impl Drop for ExecutionContext<'_> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_sys::delete_context(self.inner);
            }
        }
    }
}

unsafe impl Send for ExecutionContext<'_> {}

/// Runtime (real mode)
pub struct Runtime<'a> {
    inner: *mut std::ffi::c_void,
    _logger: &'a Logger,
}

impl<'a> Runtime<'a> {
    #[cfg(not(feature = "link_tensorrt_rtx"))]
    #[cfg(not(feature = "dlopen_tensorrt_rtx"))]
    pub fn new(logger: &'a Logger) -> Result<Self> {
        Err(Error::TrtRtxLibraryNotLoaded)
    }

    #[cfg(any(feature = "link_tensorrt_rtx", feature = "dlopen_tensorrt_rtx"))]
    pub fn new(logger: &'a Logger) -> Result<Self> {
        let logger_ptr = logger.as_logger_ptr();
        let runtime_ptr = {
            #[cfg(feature = "link_tensorrt_rtx")]
            unsafe {
                trtx_sys::create_infer_builder(logger_ptr)
            }
            #[cfg(not(feature = "link_tensorrt_rtx"))]
            #[cfg(feature = "dlopen_tensorrt_rtx")]
            unsafe {
                use libloading::Symbol;
                use std::ffi::c_void;

                use crate::TRTLIB;

                let lock = TRTLIB.read()?;
                let create_infer_builder: Symbol<fn(*mut c_void, u32) -> *mut c_void> = lock
                    .as_ref()
                    .ok_or(Error::TrtRtxLibraryNotLoaded)?
                    .get(b"createInferRuntime_INTERNAL")?;
                create_infer_builder(logger_ptr, trtx_sys::get_tensorrt_version())
            }
        };
        if runtime_ptr.is_null() {
            return Err(Error::Runtime("Failed to create runtime".to_string()));
        }
        Ok(Runtime {
            inner: runtime_ptr,
            _logger: logger,
        })
    }

    pub fn deserialize_cuda_engine(&self, data: &[u8]) -> Result<CudaEngine> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid runtime".to_string()));
        }
        let engine_ptr = unsafe {
            trtx_sys::runtime_deserialize_cuda_engine(
                self.inner,
                data.as_ptr() as *const _,
                data.len(),
            )
        };
        if engine_ptr.is_null() {
            return Err(Error::Runtime("Failed to deserialize engine".to_string()));
        }
        Ok(CudaEngine {
            inner: engine_ptr as *mut _,
        })
    }
}

impl Drop for Runtime<'_> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_sys::delete_runtime(self.inner);
            }
        }
    }
}

unsafe impl Send for Runtime<'_> {}
