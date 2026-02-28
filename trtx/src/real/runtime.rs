//! Real TensorRT runtime implementation

use cxx::UniquePtr;
use trtx_sys::nvinfer1;

use crate::error::{Error, Result};
use crate::logger::Logger;
pub use crate::real::cuda_engine::CudaEngine;
pub use crate::real::engine_inspector::EngineInspector;

/// Execution context (real mode)
pub struct ExecutionContext<'a> {
    inner: UniquePtr<nvinfer1::IExecutionContext>,
    _engine: std::marker::PhantomData<&'a CudaEngine<'a>>,
}

impl<'a> ExecutionContext<'a> {
    pub(crate) unsafe fn from_ptr(
        execution_context: *mut nvinfer1::IExecutionContext,
    ) -> Result<Self> {
        if !cfg!(feature = "mock") {
            return Err(Error::Runtime(
                "Failed to create ExecutionContext".to_string(),
            ));
        }
        Ok(ExecutionContext {
            inner: UniquePtr::from_raw(execution_context),
            _engine: Default::default(),
        })
    }

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
        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid execution context".to_string()));
            }
            let name_cstr = std::ffi::CString::new(name)?;
            let success = self
                .inner
                .pin_mut()
                .setTensorAddress(name_cstr.as_ptr(), data as *mut _);
            if !success {
                return Err(Error::Runtime("Failed to set tensor address".to_string()));
            }
        }
        Ok(())
    }

    /// Enqueues inference on the given CUDA stream.
    ///
    /// # Safety
    /// `cuda_stream` must be a valid CUDA stream, and all tensor addresses must
    /// point to valid device memory.
    pub unsafe fn enqueue_v3(&mut self, cuda_stream: *mut std::ffi::c_void) -> Result<()> {
        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid execution context".to_string()));
            }
            let success = self.inner.pin_mut().enqueueV3(cuda_stream as *mut _);
            if !success {
                return Err(Error::Runtime("Failed to enqueue inference".to_string()));
            }
        }
        Ok(())
    }
}

/// Runtime (real mode)
pub struct Runtime<'a> {
    inner: UniquePtr<nvinfer1::IRuntime>,
    _logger: &'a Logger,
}

impl<'runtime> Runtime<'runtime> {
    #[cfg(not(feature = "link_tensorrt_rtx"))]
    #[cfg(not(feature = "dlopen_tensorrt_rtx"))]
    pub fn new(logger: &'runtime Logger) -> Result<Self> {
        Err(Error::TrtRtxLibraryNotLoaded)
    }

    #[cfg(any(feature = "link_tensorrt_rtx", feature = "dlopen_tensorrt_rtx"))]
    pub fn new(logger: &'runtime Logger) -> Result<Self> {
        #[cfg(not(feature = "mock"))]
        {
            let logger_ptr = logger.as_logger_ptr();
            let runtime_ptr = {
                #[cfg(feature = "link_tensorrt_rtx")]
                unsafe {
                    trtx_sys::create_infer_runtime(logger_ptr)
                }
                #[cfg(not(feature = "link_tensorrt_rtx"))]
                #[cfg(feature = "dlopen_tensorrt_rtx")]
                unsafe {
                    use libloading::Symbol;
                    use std::ffi::c_void;

                    use crate::TRTLIB;
                    if !TRTLIB.read()?.is_some() {
                        crate::dynamically_load_tensorrt(None::<String>)?;
                    }

                    let lock = TRTLIB.read()?;
                    let create_infer_runtime: Symbol<fn(*mut c_void, u32) -> *mut c_void> = lock
                        .as_ref()
                        .ok_or(Error::TrtRtxLibraryNotLoaded)?
                        .get(b"createInferRuntime_INTERNAL")?;
                    create_infer_runtime(logger_ptr, trtx_sys::get_tensorrt_version())
                }
            } as *mut nvinfer1::IRuntime;
            if runtime_ptr.is_null() {
                return Err(Error::Runtime("Failed to create runtime".to_string()));
            }
            Ok(Runtime {
                inner: unsafe { UniquePtr::from_raw(runtime_ptr) },
                _logger: logger,
            })
        }
        #[cfg(feature = "mock")]
        Ok(Runtime {
            inner: UniquePtr::null(),
            _logger: logger,
        })
    }

    pub fn deserialize_cuda_engine(&'_ mut self, data: &[u8]) -> Result<CudaEngine<'runtime>> {
        unsafe {
            let engine = self.inner.pin_mut().deserializeCudaEngine(
                data.as_ref().as_ptr() as *const autocxx::c_void,
                data.len(),
            );
            Ok(CudaEngine::from_ptr(engine.as_mut().ok_or_else(|| {
                Error::Runtime("Failed to deserialize engine".to_string())
            })?))
        }
    }
}
