//! Runtime for deserializing and managing TensorRT engines

use crate::error::{Error, Result};
use crate::logger::Logger;
use std::ffi::CStr;

/// A CUDA engine containing optimized inference code
pub struct CudaEngine {
    #[cfg(not(feature = "mock"))]
    inner: *mut std::ffi::c_void,
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxCudaEngine,
}

impl CudaEngine {
    /// Get the number of I/O tensors
    pub fn get_nb_io_tensors(&self) -> Result<i32> {
        #[cfg(feature = "mock")]
        {
            let mut count: i32 = 0;

            let result =
                unsafe { trtx_sys::trtx_cuda_engine_get_nb_io_tensors(self.inner, &mut count) };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &[]));
            }

            Ok(count)
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid engine".to_string()));
            }

            // Use autocxx Pin to call getNbIOTensors directly
            let count = unsafe {
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ICudaEngine>(self.inner)
                    .getNbIOTensors()
            };

            Ok(count)
        }
    }

    /// Get the name of a tensor by index
    pub fn get_tensor_name(&self, index: i32) -> Result<String> {
        #[cfg(feature = "mock")]
        {
            let mut name_ptr: *const i8 = std::ptr::null();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_cuda_engine_get_tensor_name(
                    self.inner,
                    index,
                    &mut name_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            let name = unsafe { CStr::from_ptr(name_ptr) }.to_str()?.to_string();

            Ok(name)
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid engine".to_string()));
            }

            // Use autocxx Pin to call getIOTensorName directly
            let name_ptr = unsafe {
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ICudaEngine>(self.inner)
                    .getIOTensorName(index)
            };

            if name_ptr.is_null() {
                return Err(Error::InvalidArgument("Invalid tensor index".to_string()));
            }

            let name = unsafe { CStr::from_ptr(name_ptr) }.to_str()?.to_string();

            Ok(name)
        }
    }

    /// Get the shape of a tensor by name
    pub fn get_tensor_shape(&self, name: &str) -> Result<Vec<i64>> {
        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid engine".to_string()));
            }

            let name_cstr = std::ffi::CString::new(name)?;

            // Use autocxx Pin to call getTensorShape directly
            let dims = unsafe {
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::ICudaEngine>(self.inner)
                    .getTensorShape(name_cstr.as_ptr())
            };

            // Extract dimensions from Dims structure
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

        #[cfg(feature = "mock")]
        {
            // Mock implementation: return a reasonable default shape
            // In real usage, this would query the actual engine
            Ok(vec![1, 1000])
        }
    }

    /// Create an execution context for inference
    pub fn create_execution_context(&self) -> Result<ExecutionContext<'_>> {
        #[cfg(feature = "mock")]
        {
            let mut context_ptr: *mut trtx_sys::TrtxExecutionContext = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_cuda_engine_create_execution_context(
                    self.inner,
                    &mut context_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(ExecutionContext {
                inner: context_ptr,
                _engine: std::marker::PhantomData,
            })
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid engine".to_string()));
            }

            // Use autocxx Pin to call createExecutionContext directly
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
    }

    #[cfg(not(feature = "mock"))]
    #[allow(dead_code)]
    pub(crate) fn as_ptr(&self) -> *const std::ffi::c_void {
        self.inner
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            #[cfg(feature = "mock")]
            unsafe {
                trtx_sys::trtx_cuda_engine_destroy(self.inner);
            }
            #[cfg(not(feature = "mock"))]
            unsafe {
                trtx_sys::delete_engine(self.inner);
            }
        }
    }
}

unsafe impl Send for CudaEngine {}
unsafe impl Sync for CudaEngine {}

/// Execution context for running inference
pub struct ExecutionContext<'a> {
    #[cfg(not(feature = "mock"))]
    inner: *mut std::ffi::c_void,
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxExecutionContext,
    _engine: std::marker::PhantomData<&'a CudaEngine>,
}

impl<'a> ExecutionContext<'a> {
    /// Set the address of a tensor for input or output
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `data` points to valid CUDA device memory
    /// - The memory remains valid for the lifetime of inference
    /// - The memory is large enough for the tensor's size
    pub unsafe fn set_tensor_address(
        &mut self,
        name: &str,
        data: *mut std::ffi::c_void,
    ) -> Result<()> {
        #[cfg(feature = "mock")]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            let mut error_msg = [0i8; 1024];

            let result = trtx_sys::trtx_execution_context_set_tensor_address(
                self.inner,
                name_cstr.as_ptr(),
                data,
                error_msg.as_mut_ptr(),
                error_msg.len(),
            );

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(())
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid execution context".to_string()));
            }

            let name_cstr = std::ffi::CString::new(name)?;

            // Use autocxx Pin to call setTensorAddress directly
            let success = unsafe {
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::IExecutionContext>(
                    self.inner,
                )
                .setTensorAddress(name_cstr.as_ptr(), data as *mut _)
            };

            if !success {
                return Err(Error::Runtime("Failed to set tensor address".to_string()));
            }

            Ok(())
        }
    }

    /// Enqueue inference work on a CUDA stream
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `cuda_stream` is a valid CUDA stream handle (or null for default stream)
    /// - All tensor addresses have been set
    /// - CUDA context is properly initialized
    pub unsafe fn enqueue_v3(&mut self, cuda_stream: *mut std::ffi::c_void) -> Result<()> {
        #[cfg(feature = "mock")]
        {
            let mut error_msg = [0i8; 1024];

            let result = trtx_sys::trtx_execution_context_enqueue_v3(
                self.inner,
                cuda_stream,
                error_msg.as_mut_ptr(),
                error_msg.len(),
            );

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(())
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid execution context".to_string()));
            }

            // Use autocxx Pin to call enqueueV3 directly
            let success = unsafe {
                crate::autocxx_helpers::cast_and_pin::<trtx_sys::nvinfer1::IExecutionContext>(
                    self.inner,
                )
                .enqueueV3(cuda_stream as *mut _)
            };

            if !success {
                return Err(Error::Runtime("Failed to enqueue inference".to_string()));
            }

            Ok(())
        }
    }
}

impl Drop for ExecutionContext<'_> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            #[cfg(feature = "mock")]
            unsafe {
                trtx_sys::trtx_execution_context_destroy(self.inner);
            }
            #[cfg(not(feature = "mock"))]
            unsafe {
                trtx_sys::delete_context(self.inner);
            }
        }
    }
}

unsafe impl Send for ExecutionContext<'_> {}

/// Runtime for deserializing engines
pub struct Runtime<'a> {
    #[cfg(not(feature = "mock"))]
    inner: *mut std::ffi::c_void,
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxRuntime,
    _logger: &'a Logger,
}

impl<'a> Runtime<'a> {
    /// Create a new runtime
    pub fn new(logger: &'a Logger) -> Result<Self> {
        #[cfg(feature = "mock")]
        {
            let mut runtime_ptr: *mut trtx_sys::TrtxRuntime = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_runtime_create(
                    logger.as_ptr(),
                    &mut runtime_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(Runtime {
                inner: runtime_ptr,
                _logger: logger,
            })
        }

        #[cfg(not(feature = "mock"))]
        {
            let logger_ptr = logger.as_logger_ptr();
            let runtime_ptr = unsafe { trtx_sys::create_infer_runtime(logger_ptr) };

            if runtime_ptr.is_null() {
                return Err(Error::Runtime("Failed to create runtime".to_string()));
            }

            Ok(Runtime {
                inner: runtime_ptr,
                _logger: logger,
            })
        }
    }

    /// Deserialize a CUDA engine from serialized data
    pub fn deserialize_cuda_engine(&self, data: &[u8]) -> Result<CudaEngine> {
        #[cfg(feature = "mock")]
        {
            let mut engine_ptr: *mut trtx_sys::TrtxCudaEngine = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_runtime_deserialize_cuda_engine(
                    self.inner,
                    data.as_ptr() as *const std::ffi::c_void,
                    data.len(),
                    &mut engine_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(CudaEngine { inner: engine_ptr })
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid runtime".to_string()));
            }

            // Use wrapper for now - autocxx::c_ulonglong type not accessible from trtx crate
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
}

impl Drop for Runtime<'_> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            #[cfg(feature = "mock")]
            unsafe {
                trtx_sys::trtx_runtime_destroy(self.inner);
            }
            #[cfg(not(feature = "mock"))]
            unsafe {
                trtx_sys::delete_runtime(self.inner);
            }
        }
    }
}

unsafe impl Send for Runtime<'_> {}
