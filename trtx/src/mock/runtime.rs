//! Mock runtime implementations

use crate::error::Result;
use crate::logger::Logger;
use std::ffi::CStr;

use super::from_ffi;

/// CUDA engine (mock mode)
pub struct CudaEngine {
    pub(crate) inner: *mut trtx_sys::TrtxCudaEngine,
}

impl CudaEngine {
    pub(crate) fn from_mock_ptr(ptr: *mut trtx_sys::TrtxCudaEngine) -> Self {
        CudaEngine { inner: ptr }
    }

    pub fn get_nb_io_tensors(&self) -> Result<i32> {
        get_nb_io_tensors(self.inner)
    }

    pub fn get_tensor_name(&self, index: i32) -> Result<String> {
        get_tensor_name(self.inner, index)
    }

    pub fn get_tensor_shape(&self, _name: &str) -> Result<Vec<i64>> {
        Ok(crate::mock::default_engine_tensor_shape())
    }

    pub fn create_execution_context(&self) -> Result<ExecutionContext<'_>> {
        let context_ptr = create_execution_context(self.inner)?;
        Ok(ExecutionContext::from_mock_ptr(context_ptr))
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        destroy_engine(self.inner);
    }
}

unsafe impl Send for CudaEngine {}
unsafe impl Sync for CudaEngine {}

/// Execution context (mock mode)
pub struct ExecutionContext<'a> {
    inner: *mut trtx_sys::TrtxExecutionContext,
    _engine: std::marker::PhantomData<&'a CudaEngine>,
}

impl<'a> ExecutionContext<'a> {
    pub(crate) fn from_mock_ptr(ptr: *mut trtx_sys::TrtxExecutionContext) -> Self {
        ExecutionContext {
            inner: ptr,
            _engine: std::marker::PhantomData,
        }
    }

    pub unsafe fn set_tensor_address(
        &mut self,
        name: &str,
        data: *mut std::ffi::c_void,
    ) -> Result<()> {
        set_tensor_address(self.inner, name, data)
    }

    pub unsafe fn enqueue_v3(&mut self, cuda_stream: *mut std::ffi::c_void) -> Result<()> {
        enqueue_v3(self.inner, cuda_stream)
    }
}

impl Drop for ExecutionContext<'_> {
    fn drop(&mut self) {
        destroy_context(self.inner);
    }
}

unsafe impl Send for ExecutionContext<'_> {}

/// Runtime (mock mode)
pub struct Runtime<'a> {
    inner: *mut trtx_sys::TrtxRuntime,
    _logger: &'a Logger,
}

impl<'a> Runtime<'a> {
    pub fn new(logger: &'a Logger) -> Result<Self> {
        let runtime_ptr = trtx_runtime_create(logger.as_ptr())?;
        Ok(Runtime {
            inner: runtime_ptr,
            _logger: logger,
        })
    }

    pub fn deserialize_cuda_engine(&self, data: &[u8]) -> Result<CudaEngine> {
        let engine_ptr = deserialize_cuda_engine(self.inner, data)?;
        Ok(CudaEngine::from_mock_ptr(engine_ptr))
    }
}

impl Drop for Runtime<'_> {
    fn drop(&mut self) {
        destroy_runtime(self.inner);
    }
}

unsafe impl Send for Runtime<'_> {}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

fn get_nb_io_tensors(engine_ptr: *mut trtx_sys::TrtxCudaEngine) -> Result<i32> {
    let mut count: i32 = 0;

    let result = unsafe { trtx_sys::trtx_cuda_engine_get_nb_io_tensors(engine_ptr, &mut count) };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &[]));
    }

    Ok(count)
}

fn get_tensor_name(engine_ptr: *mut trtx_sys::TrtxCudaEngine, index: i32) -> Result<String> {
    let mut name_ptr: *const i8 = std::ptr::null();
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_cuda_engine_get_tensor_name(
            engine_ptr,
            index,
            &mut name_ptr,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }

    let name = unsafe { CStr::from_ptr(name_ptr) }.to_str()?.to_string();

    Ok(name)
}

fn trtx_runtime_create(
    logger_ptr: *mut trtx_sys::TrtxLogger,
) -> Result<*mut trtx_sys::TrtxRuntime> {
    let mut runtime_ptr: *mut trtx_sys::TrtxRuntime = std::ptr::null_mut();
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_runtime_create(
            logger_ptr,
            &mut runtime_ptr,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }

    Ok(runtime_ptr)
}

fn deserialize_cuda_engine(
    runtime_ptr: *mut trtx_sys::TrtxRuntime,
    data: &[u8],
) -> Result<*mut trtx_sys::TrtxCudaEngine> {
    let mut engine_ptr: *mut trtx_sys::TrtxCudaEngine = std::ptr::null_mut();
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_runtime_deserialize_cuda_engine(
            runtime_ptr,
            data.as_ptr() as *const std::ffi::c_void,
            data.len(),
            &mut engine_ptr,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }

    Ok(engine_ptr)
}

fn create_execution_context(
    engine_ptr: *mut trtx_sys::TrtxCudaEngine,
) -> Result<*mut trtx_sys::TrtxExecutionContext> {
    let mut context_ptr: *mut trtx_sys::TrtxExecutionContext = std::ptr::null_mut();
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_cuda_engine_create_execution_context(
            engine_ptr,
            &mut context_ptr,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }

    Ok(context_ptr)
}

fn set_tensor_address(
    context_ptr: *mut trtx_sys::TrtxExecutionContext,
    name: &str,
    data: *mut std::ffi::c_void,
) -> Result<()> {
    let name_cstr = std::ffi::CString::new(name)?;
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_execution_context_set_tensor_address(
            context_ptr,
            name_cstr.as_ptr(),
            data,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }

    Ok(())
}

fn enqueue_v3(
    context_ptr: *mut trtx_sys::TrtxExecutionContext,
    cuda_stream: *mut std::ffi::c_void,
) -> Result<()> {
    let mut error_msg = [0i8; 1024];

    let result = unsafe {
        trtx_sys::trtx_execution_context_enqueue_v3(
            context_ptr,
            cuda_stream,
            error_msg.as_mut_ptr(),
            error_msg.len(),
        )
    };

    if result != trtx_sys::TRTX_SUCCESS as i32 {
        return Err(from_ffi(result, &error_msg));
    }

    Ok(())
}

fn destroy_engine(engine_ptr: *mut trtx_sys::TrtxCudaEngine) {
    if !engine_ptr.is_null() {
        unsafe {
            trtx_sys::trtx_cuda_engine_destroy(engine_ptr);
        }
    }
}

fn destroy_context(context_ptr: *mut trtx_sys::TrtxExecutionContext) {
    if !context_ptr.is_null() {
        unsafe {
            trtx_sys::trtx_execution_context_destroy(context_ptr);
        }
    }
}

fn destroy_runtime(runtime_ptr: *mut trtx_sys::TrtxRuntime) {
    if !runtime_ptr.is_null() {
        unsafe {
            trtx_sys::trtx_runtime_destroy(runtime_ptr);
        }
    }
}
