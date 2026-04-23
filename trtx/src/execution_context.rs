use std::ffi::CString;
use std::pin::Pin;

use cxx::UniquePtr;
use trtx_sys::nvinfer1;

pub use crate::cuda_engine::CudaEngine;
pub use crate::engine_inspector::EngineInspector;
use crate::error::{Error, Result};
use crate::interfaces::{DebugListener, ProcessDebugTensor, Profiler, ReportLayerTime};

/// [`trtx_sys::nvinfer1::IExecutionContext`] — C++ [`nvinfer1::IExecutionContext`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_execution_context.html).
///
/// `inner` is declared last so it is dropped first (see [`Drop`]): TensorRT must release
/// [`DebugListener`](crate::interfaces::DebugListener) / [`Profiler`](crate::interfaces::Profiler)
/// pointers before their Rust wrappers run destructors.
pub struct ExecutionContext<'a> {
    _engine: std::marker::PhantomData<&'a CudaEngine<'a>>,
    debug_listener: Option<Pin<Box<DebugListener>>>,
    profiler: Option<Pin<Box<Profiler>>>,
    inner: UniquePtr<nvinfer1::IExecutionContext>,
}

impl<'a> ExecutionContext<'a> {
    pub(crate) unsafe fn from_ptr(
        execution_context: *mut nvinfer1::IExecutionContext,
    ) -> Result<Self> {
        #[cfg(not(feature = "mock_runtime"))]
        if execution_context.is_null() {
            return Err(Error::Runtime(
                "Failed to create ExecutionContext".to_string(),
            ));
        }
        Ok(ExecutionContext {
            _engine: Default::default(),
            debug_listener: None,
            profiler: None,
            inner: UniquePtr::from_raw(execution_context),
        })
    }

    /// See [nvinfer1::IExecutionContext::setProfiler].
    ///
    /// Only one profiler may be set for the lifetime of this context (same restriction as
    /// [`Self::set_debug_listener`]).
    pub fn set_profiler(&mut self, profiler: Box<dyn ReportLayerTime>) -> Result<()> {
        let profiler = Profiler::new(profiler)?;
        if self.profiler.is_some() {
            panic!("Setting a profiler more than once not supported at the moment");
        }
        self.profiler = Some(profiler);
        #[cfg(not(feature = "mock_runtime"))]
        {
            if !self.inner.is_null() {
                unsafe {
                    self.inner.pin_mut().setProfiler(
                        self.profiler
                            .as_ref()
                            .expect("profiler can't be empty, we just set it")
                            .as_raw(),
                    );
                }
            }
        }
        Ok(())
    }

    /// See [nvinfer1::IExecutionContext::reportToProfiler].
    ///
    /// When enqueue does not emit the profile (see C++ `setEnqueueEmitsProfile(false)`), call this
    /// after [`Self::enqueue_v3`] (or after a captured CUDA graph launch) while the same stream is
    /// still valid.
    pub fn report_to_profiler(&self) -> Result<bool> {
        #[cfg(not(feature = "mock_runtime"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid execution context".to_string()));
            }
            Ok(self.inner.reportToProfiler())
        }
        #[cfg(feature = "mock_runtime")]
        {
            Ok(true)
        }
    }

    /// See [nvinfer1::IExecutionContext::setDebugListener].
    /// The Rust bindings only allow setting the debug listener once per execution context.
    pub fn set_debug_listener(&mut self, listener: Box<dyn ProcessDebugTensor>) -> Result<()> {
        let debug_listener = DebugListener::new(listener)?;
        if self.debug_listener.is_some() {
            panic!("Setting a debug listener more than once not supported at the moment");
        }
        self.debug_listener = Some(debug_listener);
        #[cfg(not(feature = "mock_runtime"))]
        {
            let success = unsafe {
                self.inner.pin_mut().setDebugListener(
                    self.debug_listener
                        .as_ref()
                        .expect("debug_listener can't be empty, we just set it")
                        .as_raw(),
                )
            };
            if !success {
                self.debug_listener = None;
                return Err(Error::Runtime("setDebugListener failed".to_string()));
            }
        }
        Ok(())
    }

    /// See [nvinfer1::IExecutionContext::setTensorDebugState].
    pub fn set_tensor_debug_state(&mut self, name: &str, flag: bool) -> Result<()> {
        let name = CString::new(name)?;
        if !unsafe {
            self.inner
                .pin_mut()
                .setTensorDebugState(name.as_ptr(), flag)
        } {
            Err(Error::FailedToSetProperty(
                crate::error::PropertySetAttempt::ExecutionContextTensorDebugState,
            ))
        } else {
            Ok(())
        }
    }

    /// See [nvinfer1::IExecutionContext::getDebugState].
    pub fn get_tensor_debug_state(&self, name: &str) -> Result<bool> {
        let name = CString::new(name)?;
        unsafe { Ok(self.inner.getDebugState(name.as_ptr())) }
    }

    /// See [nvinfer1::IExecutionContext::setAllTensorsDebugState].
    pub fn set_all_tensors_debug_state(&mut self, flag: bool) -> Result<()> {
        if !self.inner.pin_mut().setAllTensorsDebugState(flag) {
            Err(Error::FailedToSetProperty(
                crate::error::PropertySetAttempt::ExecutionContextTensorDebugState,
            ))
        } else {
            Ok(())
        }
    }
    /// See [nvinfer1::IExecutionContext::setUnfusedTensorsDebugState].
    pub fn set_unfused_tensors_debug_state(&mut self, flag: bool) -> Result<()> {
        if !self.inner.pin_mut().setUnfusedTensorsDebugState(flag) {
            Err(Error::FailedToSetProperty(
                crate::error::PropertySetAttempt::ExecutionContextTensorDebugState,
            ))
        } else {
            Ok(())
        }
    }
    /// See [nvinfer1::IExecutionContext::getUnfusedTensorsDebugState].
    pub fn get_unfused_tensor_debug_state(&self) -> bool {
        self.inner.getUnfusedTensorsDebugState()
    }

    /// Binds a tensor to a device memory address.
    ///
    /// # Safety
    /// `data` must point to valid CUDA memory with at least the tensor's size in bytes,
    /// and remain valid for the duration of inference.
    ///
    /// See [nvinfer1::IExecutionContext::setTensorAddress]
    pub unsafe fn set_tensor_address(
        &mut self,
        name: &str,
        data: *mut std::ffi::c_void,
    ) -> Result<()> {
        #[cfg(not(feature = "mock_runtime"))]
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
        #[cfg(not(feature = "mock_runtime"))]
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
