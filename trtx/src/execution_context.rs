use std::ffi::{CStr, CString};
use std::pin::Pin;

use cxx::UniquePtr;
use trtx_sys::nvinfer1;

pub use crate::cuda_engine::CudaEngine;
pub use crate::engine_inspector::EngineInspector;
use crate::error::{Error, PropertySetAttempt, Result};
use crate::interfaces::{
    DebugListener, ErrorRecorder, ProcessDebugTensor, Profiler, RecordError, ReportLayerTime,
};

/// [`trtx_sys::nvinfer1::IExecutionContext`] — C++ [`nvinfer1::IExecutionContext`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_execution_context.html).
///
/// `inner` is declared last so it is dropped first (see [`Drop`]): TensorRT must release
/// [`DebugListener`](crate::interfaces::DebugListener) / [`Profiler`](crate::interfaces::Profiler)
/// pointers before their Rust wrappers run destructors.
pub struct ExecutionContext<'a> {
    _engine: std::marker::PhantomData<&'a CudaEngine<'a>>,
    debug_listener: Option<Pin<Box<DebugListener>>>,
    profiler: Option<Pin<Box<Profiler>>>,
    error_recorder: Option<Pin<Box<ErrorRecorder>>>,
    inner: UniquePtr<nvinfer1::IExecutionContext>,
}

impl std::fmt::Debug for ExecutionContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("inner", &format!("{:x}", self.inner.as_ptr() as usize))
            .finish_non_exhaustive()
    }
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
            error_recorder: None,
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

    /// See [nvinfer1::IExecutionContext::setDebugSync].
    pub fn set_debug_sync(&mut self, sync: bool) {
        self.inner.pin_mut().setDebugSync(sync);
    }

    /// See [nvinfer1::IExecutionContext::getDebugSync].
    pub fn debug_sync(&self) -> bool {
        self.inner.getDebugSync()
    }

    // omitted

    ///// See [nvinfer1::IExecutionContext::getProfiler].
    //pub fn profiler_ptr(&self) -> *mut nvinfer1::IProfiler {
    //self.inner.getProfiler()
    //}

    ///// See [nvinfer1::IExecutionContext::getEngine].
    //pub fn engine_ptr(&self) -> *const nvinfer1::ICudaEngine {
    //self.inner.getEngine() as *const nvinfer1::ICudaEngine
    //}

    ///// See [nvinfer1::IExecutionContext::getDebugListener].
    //pub fn debug_listener_ptr(&mut self) -> *mut nvinfer1::IDebugListener {
    //self.inner.pin_mut().getDebugListener()
    //}

    /// See [nvinfer1::IExecutionContext::setName].
    pub fn set_name(&mut self, name: &str) -> Result<()> {
        let name = CString::new(name)?;
        unsafe { self.inner.pin_mut().setName(name.as_ptr()) };
        Ok(())
    }

    /// See [nvinfer1::IExecutionContext::getName].
    pub fn name(&self) -> Result<String> {
        let ptr = self.inner.getName();
        if ptr.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string())
    }

    /// See [nvinfer1::IExecutionContext::setDeviceMemory].
    /// See [ExecutionContext::set_device_memory_v2].
    ///
    /// # Safety
    ///
    /// `memory` must be a pointer to a valid CUDA device memory and follow CUDA semantics.
    /// It must be sufficiently larger for the needs of the engine
    pub unsafe fn set_device_memory(&mut self, memory: *mut std::ffi::c_void) {
        unsafe { self.inner.pin_mut().setDeviceMemory(memory as *mut _) };
    }

    /// See [nvinfer1::IExecutionContext::setDeviceMemoryV2].
    ///
    /// # Safety
    ///
    /// `memory` must be a pointer to a valid CUDA device memory and follow CUDA semantics
    /// `size` must be the correct size that is allocated for `memory`
    pub unsafe fn set_device_memory_v2(&mut self, memory: *mut std::ffi::c_void, size: i64) {
        unsafe {
            self.inner
                .pin_mut()
                .setDeviceMemoryV2(memory as *mut _, size)
        };
    }

    /// See [nvinfer1::IExecutionContext::getTensorStrides].
    pub fn tensor_strides(&self, name: &str) -> Result<nvinfer1::Dims64> {
        let name = CString::new(name)?;
        Ok(unsafe { self.inner.getTensorStrides(name.as_ptr()) })
    }

    /// See [nvinfer1::IExecutionContext::getOptimizationProfile].
    pub fn optimization_profile(&self) -> i32 {
        self.inner.getOptimizationProfile()
    }

    /// See [nvinfer1::IExecutionContext::setInputShape].
    pub fn set_input_shape(&mut self, name: &str, dims: &nvinfer1::Dims64) -> Result<()> {
        let name = CString::new(name)?;
        if unsafe { self.inner.pin_mut().setInputShape(name.as_ptr(), dims) } {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(
                PropertySetAttempt::ExecutionContextInputShape,
            ))
        }
    }

    /// See [nvinfer1::IExecutionContext::getTensorShape].
    pub fn tensor_shape(&self, name: &str) -> Result<nvinfer1::Dims64> {
        let name = CString::new(name)?;
        Ok(unsafe { self.inner.getTensorShape(name.as_ptr()) })
    }

    /// See [nvinfer1::IExecutionContext::allInputDimensionsSpecified].
    pub fn all_input_dimensions_specified(&self) -> bool {
        self.inner.allInputDimensionsSpecified()
    }

    /// See [nvinfer1::IExecutionContext::setErrorRecorder].
    pub fn set_error_recorder(&mut self, error_recorder: Box<dyn RecordError>) -> Result<()> {
        let error_recorder = ErrorRecorder::new(error_recorder)?;
        if self.error_recorder.is_some() {
            // would need to make sure that we don't destroy a monitor still in use
            // could offer this as an unsafe method for users who only set this when there is no
            // build process active. Or we only accept a ref to progress monitor and force user
            // via lifetimes to keep this alive for builder config lifetime
            panic!("Setting a progress monitor more than once not supported at the moment");
        }
        self.error_recorder = Some(error_recorder);
        let rec = self
            .error_recorder
            .as_mut()
            .unwrap()
            .as_trt_error_recorder();
        #[cfg(not(feature = "mock"))]
        unsafe {
            self.inner.pin_mut().setErrorRecorder(rec)
        };
        Ok(())
    }

    /// See [nvinfer1::IExecutionContext::setOptimizationProfileAsync].
    ///
    /// # Safety
    ///
    /// `stream` must be a pointer to a valid CUDA stream and follow CUDA semantics
    pub unsafe fn set_optimization_profile_async(
        &mut self,
        profile_index: i32,
        stream: *mut std::ffi::c_void,
    ) -> bool {
        unsafe {
            self.inner
                .pin_mut()
                .setOptimizationProfileAsync(profile_index, stream as *mut _)
        }
    }

    /// See [nvinfer1::IExecutionContext::setEnqueueEmitsProfile].
    pub fn set_enqueue_emits_profile(&mut self, enqueue_emits_profile: bool) {
        self.inner
            .pin_mut()
            .setEnqueueEmitsProfile(enqueue_emits_profile);
    }

    /// See [nvinfer1::IExecutionContext::getEnqueueEmitsProfile].
    pub fn enqueue_emits_profile(&self) -> bool {
        self.inner.getEnqueueEmitsProfile()
    }

    /// See [nvinfer1::IExecutionContext::reportToProfiler].
    ///
    /// When enqueue does not emit the profile (see C++ `setEnqueueEmitsProfile(false)`), call this
    /// after [`Self::enqueue_v3`] (or after a captured CUDA graph launch) while the same stream is
    /// still valid.
    pub fn report_to_profiler(&self) -> Result<()> {
        #[cfg(not(feature = "mock_runtime"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid execution context".to_string()));
            }
            if self.inner.reportToProfiler() {
                Ok(())
            } else {
                Err(Error::FailedToReportToProfiler)
            }
        }
        #[cfg(feature = "mock_runtime")]
        {
            Ok(())
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
    pub fn tensor_debug_state(&self, name: &str) -> Result<bool> {
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
    pub fn unfused_tensor_debug_state(&self) -> bool {
        self.inner.getUnfusedTensorsDebugState()
    }

    /// See [nvinfer1::IExecutionContext::getRuntimeConfig].
    pub fn runtime_config(&self) -> *mut nvinfer1::IRuntimeConfig {
        self.inner.getRuntimeConfig()
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

    /// See [nvinfer1::IExecutionContext::getTensorAddress].
    pub fn tensor_address(&self, name: &str) -> Result<*const std::ffi::c_void> {
        let name = CString::new(name)?;
        Ok(unsafe { self.inner.getTensorAddress(name.as_ptr()) as *const std::ffi::c_void })
    }

    /// See [nvinfer1::IExecutionContext::setOutputTensorAddress].
    ///
    /// # Safety
    /// `data` must point to valid CUDA memory with at least the tensor's size in bytes,
    /// and remain valid for the duration of inference.
    pub unsafe fn set_output_tensor_address(
        &mut self,
        name: &str,
        data: *mut std::ffi::c_void,
    ) -> Result<bool> {
        let name = CString::new(name)?;
        Ok(unsafe {
            self.inner
                .pin_mut()
                .setOutputTensorAddress(name.as_ptr(), data as *mut _)
        })
    }

    /// See [nvinfer1::IExecutionContext::setInputTensorAddress].
    ///
    /// # Safety
    /// `data` must point to valid CUDA memory with at least the tensor's size in bytes,
    /// and remain valid for the duration of inference.
    pub unsafe fn set_input_tensor_address(
        &mut self,
        name: &str,
        data: *const std::ffi::c_void,
    ) -> Result<bool> {
        let name = CString::new(name)?;
        Ok(unsafe {
            self.inner
                .pin_mut()
                .setInputTensorAddress(name.as_ptr(), data as *const _)
        })
    }

    /// See [nvinfer1::IExecutionContext::getOutputTensorAddress].
    pub fn output_tensor_address(&self, name: &str) -> Result<*mut std::ffi::c_void> {
        let name = CString::new(name)?;
        Ok(unsafe { self.inner.getOutputTensorAddress(name.as_ptr()) as *mut std::ffi::c_void })
    }

    /// See [nvinfer1::IExecutionContext::updateDeviceMemorySizeForShapes].
    pub fn update_device_memory_size_for_shapes(&mut self) -> usize {
        self.inner.pin_mut().updateDeviceMemorySizeForShapes()
    }

    /// See [nvinfer1::IExecutionContext::setInputConsumedEvent].
    ///
    /// # Safety
    /// `event` must point to valid CUDA event and usage must follow CUDA semantics.
    pub unsafe fn set_input_consumed_event(&mut self, event: *mut std::ffi::c_void) -> bool {
        unsafe { self.inner.pin_mut().setInputConsumedEvent(event as *mut _) }
    }

    /// See [nvinfer1::IExecutionContext::getInputConsumedEvent].
    pub fn input_consumed_event(&self) -> *mut std::ffi::c_void {
        self.inner.getInputConsumedEvent() as *mut std::ffi::c_void
    }

    ///// See [nvinfer1::IExecutionContext::setOutputAllocator].
    //pub unsafe fn set_output_allocator(
    //&mut self,
    //name: &str,
    //allocator: *mut nvinfer1::IOutputAllocator,
    //) -> Result<bool> {
    //let name = CString::new(name)?;
    //Ok(unsafe {
    //self.inner
    //.pin_mut()
    //.setOutputAllocator(name.as_ptr(), allocator)
    //})
    //}

    ///// See [nvinfer1::IExecutionContext::getOutputAllocator].
    //pub fn output_allocator(&self, name: &str) -> Result<*mut nvinfer1::IOutputAllocator> {
    //let name = CString::new(name)?;
    //Ok(unsafe { self.inner.getOutputAllocator(name.as_ptr()) })
    //}

    /// See [nvinfer1::IExecutionContext::getMaxOutputSize].
    pub fn max_output_size(&self, name: &str) -> Result<i64> {
        let name = CString::new(name)?;
        Ok(unsafe { self.inner.getMaxOutputSize(name.as_ptr()) })
    }

    ///// See [nvinfer1::IExecutionContext::setTemporaryStorageAllocator].
    //pub unsafe fn set_temporary_storage_allocator(
    //&mut self,
    //allocator: *mut nvinfer1::IGpuAllocator,
    //) -> bool {
    //unsafe { self.inner.pin_mut().setTemporaryStorageAllocator(allocator) }
    //}

    ///// See [nvinfer1::IExecutionContext::getTemporaryStorageAllocator].
    //pub fn temporary_storage_allocator(&self) -> *mut nvinfer1::IGpuAllocator {
    //self.inner.getTemporaryStorageAllocator()
    //}

    /// See [nvinfer1::IExecutionContext::setPersistentCacheLimit].
    pub fn set_persistent_cache_limit(&mut self, size: usize) {
        self.inner.pin_mut().setPersistentCacheLimit(size);
    }

    /// See [nvinfer1::IExecutionContext::getPersistentCacheLimit].
    pub fn persistent_cache_limit(&self) -> usize {
        self.inner.getPersistentCacheLimit()
    }

    /// See [nvinfer1::IExecutionContext::setNvtxVerbosity].
    pub fn set_nvtx_verbosity(&mut self, verbosity: nvinfer1::ProfilingVerbosity) -> bool {
        self.inner.pin_mut().setNvtxVerbosity(verbosity)
    }

    /// See [nvinfer1::IExecutionContext::getNvtxVerbosity].
    pub fn nvtx_verbosity(&self) -> nvinfer1::ProfilingVerbosity {
        self.inner.getNvtxVerbosity()
    }

    /// See [nvinfer1::IExecutionContext::setAuxStreams].
    ///
    /// # Safety
    /// `aux_streams` must be valid CUDA streams and follow CUDA semantics
    pub unsafe fn set_aux_streams(&mut self, aux_streams: &mut [*mut std::ffi::c_void]) {
        unsafe {
            self.inner.pin_mut().setAuxStreams(
                aux_streams.as_mut_ptr() as *mut *mut _,
                aux_streams.len().try_into().unwrap(),
            )
        };
    }

    /// Enqueues inference on the given CUDA stream.
    ///
    /// # Safety
    /// `cuda_stream` must be a valid CUDA stream, and all tensor addresses must
    /// point to valid device memory.
    ///
    /// See [nvinfer1::IExecutionContext::enqueueV3]
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

    #[cfg(feature = "v_1_4")]
    /// See [nvinfer1::IExecutionContext::setCommunicator]
    ///
    /// # Safety
    /// `comm` must be a pointer to a valid NCCL communicator and follow valid NCCL/CUDA
    /// usage
    pub unsafe fn set_communicator(&mut self, comm: *mut std::ffi::c_void) -> Result<()> {
        if self.inner.pin_mut().setCommunicator(comm as *mut _) {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(
                crate::error::PropertySetAttempt::ExecutionContextNcclCommunicator,
            ))
        }
    }
}
