//! Rust implementations of TensorRT callback / allocator interfaces (bridged to C++).
//!
//! Versioned runtime interfaces live under `nvinfer1::v_1_0` in C++; see the
//! [annotated class list](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/annotated.html).

use crate::{Error, Result};
use cxx::UniquePtr;
use std::ptr::null_mut;
use std::{ffi::CStr, pin::Pin};
use trtx_sys::{
    nvinfer1, trtx_create_debug_listener, trtx_create_error_recorder, trtx_create_gpu_allocator,
    trtx_create_progress_monitor,
};
use trtx_sys::{DataType, Dims64, ErrorCode, TensorLocation};

/// Rust trait implemented by [`ProgressMonitor`] for [`trtx_sys::nvinfer1::IProgressMonitor`]; C++ [`nvinfer1::v_1_0::IProgressMonitor`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_progress_monitor.html).
///
/// Use with [`crate::BuilderConfig::set_progress_monitor`].
pub trait MonitorProgress: Send + Sync {
    /// See [`trtx_sys::nvinfer1::IProgressMonitor`] / C++ [`phaseStart`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_progress_monitor.html).
    fn phase_start(&self, phase_name: &str, parent_phase: Option<&str>, num_steps: i32);
    /// See [`trtx_sys::nvinfer1::IProgressMonitor`] / C++ [`stepComplete`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_progress_monitor.html). Return whether to continue building or cancel.
    fn step_complete(&self, phase_name: &str, step: i32) -> std::ops::ControlFlow<()>;
    /// See [`trtx_sys::nvinfer1::IProgressMonitor`] / C++ [`phaseFinish`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_progress_monitor.html).
    fn phase_finish(&self, phase_name: &str);
}

#[allow(non_snake_case)]
unsafe extern "system" fn ProgressMonitor_phaseStart(
    this: *mut std::ffi::c_void,
    phaseName: *const ::std::os::raw::c_char,
    parentPhase: *const ::std::os::raw::c_char,
    nbSteps: i32,
) {
    let this = this as *mut ProgressMonitor;
    let phase_name = CStr::from_ptr(phaseName);
    let parent_phase =
        (!parentPhase.is_null()).then(|| CStr::from_ptr(phaseName).to_string_lossy());
    this.as_mut().unwrap().rust_impl.phase_start(
        &phase_name.to_string_lossy(),
        parent_phase.as_deref(),
        nbSteps,
    );
}
#[allow(non_snake_case)]
unsafe extern "system" fn ProgressMonitor_stepComplete(
    this: *mut std::ffi::c_void,
    phaseName: *const ::std::os::raw::c_char,
    step: i32,
) -> bool {
    let this = this as *mut ProgressMonitor;
    let phase_name = CStr::from_ptr(phaseName);
    this.as_mut()
        .unwrap()
        .rust_impl
        .step_complete(&phase_name.to_string_lossy(), step)
        .is_continue()
}
#[allow(non_snake_case)]
unsafe extern "system" fn ProgressMonitor_phaseFinish(
    this: *mut std::ffi::c_void,
    phaseName: *const ::std::os::raw::c_char,
) {
    let this = this as *mut ProgressMonitor;
    let phase_name = CStr::from_ptr(phaseName);
    this.as_mut()
        .unwrap()
        .rust_impl
        .phase_finish(&phase_name.to_string_lossy());
}

/// Bridges to [`trtx_sys::nvinfer1::IProgressMonitor`]; C++ [`nvinfer1::v_1_0::IProgressMonitor`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_progress_monitor.html).
///
/// Construct with a [`MonitorProgress`] implementation.
#[repr(C)]
pub(crate) struct ProgressMonitor {
    cpp_obj: UniquePtr<nvinfer1::IProgressMonitor>,
    rust_impl: Box<dyn MonitorProgress>,
}

impl ProgressMonitor {
    pub(crate) fn new(inner: Box<dyn MonitorProgress>) -> Result<Pin<Box<ProgressMonitor>>> {
        let mut rust_obj = Box::pin(ProgressMonitor {
            cpp_obj: UniquePtr::null(),
            rust_impl: inner,
        });

        unsafe {
            let cpp_obj = UniquePtr::from_raw(trtx_create_progress_monitor(
                rust_obj.as_mut().get_unchecked_mut() as *mut ProgressMonitor
                    as *mut std::ffi::c_void,
                ProgressMonitor_phaseStart,
                ProgressMonitor_stepComplete,
                ProgressMonitor_phaseFinish,
            ));
            if cpp_obj.is_null() {
                return Err(Error::Runtime(
                    "Failed to allocate object for IProgressMonitor subclass".to_string(),
                ));
            }
            rust_obj.cpp_obj = cpp_obj;
        }
        Ok(rust_obj)
    }
    pub fn as_trt_progress_monitor(&self) -> *mut nvinfer1::IProgressMonitor {
        self.cpp_obj.as_mut_ptr()
    }
}

/// C callbacks for GpuAllocatorSubclass (bridge to Rust). `this` is *mut RefCell<GpuAllocator>.
#[allow(non_snake_case)]
unsafe extern "system" fn GpuAllocator_allocateAsync(
    this: *const std::ffi::c_void,
    size: u64,
    alignment: u64,
    flags: u32,
    cuda_stream: *mut std::ffi::c_void,
) -> *mut std::ffi::c_void {
    let this = this as *const GpuAllocator;
    this.as_ref()
        .unwrap()
        .rust_impl
        .allocate_async(size, alignment, flags, cuda_stream)
}
#[allow(non_snake_case)]
unsafe extern "system" fn GpuAllocator_reallocate(
    this: *const std::ffi::c_void,
    memory: *mut std::ffi::c_void,
    alignment: u64,
    new_size: u64,
) -> *mut std::ffi::c_void {
    let this = this as *const GpuAllocator;
    this.as_ref()
        .unwrap()
        .rust_impl
        .reallocate(memory, alignment, new_size)
}
#[allow(non_snake_case)]
unsafe extern "system" fn GpuAllocator_deallocateAsync(
    this: *const std::ffi::c_void,
    memory: *mut std::ffi::c_void,
    cuda_stream: *mut std::ffi::c_void,
) -> bool {
    let this = this as *const GpuAllocator;
    this.as_ref()
        .unwrap()
        .rust_impl
        .deallocate_async(memory, cuda_stream)
}

/// Bridges to [`trtx_sys::nvinfer1::IGpuAllocator`]; C++ [`nvinfer1::v_1_0::IGpuAllocator`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_gpu_allocator.html).
///
/// Construct with an [`AllocateGpu`] implementation.
#[repr(C)]
pub struct GpuAllocator {
    cpp_obj: UniquePtr<nvinfer1::IGpuAllocator>,
    rust_impl: Box<dyn AllocateGpu>,
}

impl GpuAllocator {
    pub fn new(inner: Box<dyn AllocateGpu>) -> Result<Pin<Box<Self>>> {
        let mut rust_obj = Box::pin(GpuAllocator {
            cpp_obj: UniquePtr::null(),
            rust_impl: inner,
        });
        unsafe {
            let cpp_obj = UniquePtr::from_raw(trtx_create_gpu_allocator(
                rust_obj.as_mut().get_unchecked_mut() as *mut GpuAllocator as *mut std::ffi::c_void,
                GpuAllocator_allocateAsync,
                GpuAllocator_reallocate,
                GpuAllocator_deallocateAsync,
            ));
            if cpp_obj.is_null() {
                return Err(Error::Runtime(
                    "Failed to allocate object for IGpuAllocator subclass".to_string(),
                ));
            }
            rust_obj.cpp_obj = cpp_obj;
        }
        Ok(rust_obj)
    }

    pub fn as_trt_gpu_allocator(&self) -> *mut nvinfer1::IGpuAllocator {
        self.cpp_obj.as_mut_ptr()
    }
}

/// Implemented by [`GpuAllocator`] for [`trtx_sys::nvinfer1::IGpuAllocator`]; C++ [`nvinfer1::v_1_0::IGpuAllocator`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_gpu_allocator.html).
pub trait AllocateGpu: Send + Sync {
    // we omit the following deprecated methods
    //fn allocate(&mut self, size: u64, alignment: u64, flags: u32) -> *mut autocxx::c_void;
    //unsafe fn deallocate(&mut self, data: *mut autocxx::c_void) -> bool;

    /// # Safety
    /// User needs to ensure memory safety for CUDA device pointers and follow regular CUDA rules
    unsafe fn allocate_async(
        &self,
        size: u64,
        alignment: u64,
        flags: u32,
        cuda_stream: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;
    /// # Safety
    /// User needs to ensure memory safety for CUDA device pointers and follow regular CUDA rules
    unsafe fn reallocate(
        &self,
        memory: *mut std::ffi::c_void,
        alignment: u64,
        new_size: u64,
    ) -> *mut std::ffi::c_void;
    /// # Safety
    /// User needs to ensure memory safety for CUDA device pointers and follow regular CUDA rules
    unsafe fn deallocate_async(
        &self,
        data: *mut std::ffi::c_void,
        cuda_stream: *mut std::ffi::c_void,
    ) -> bool;
}

/// C callbacks for ErrorRecorderSubclass (bridge to Rust). `this` is *mut RefCell<ErrorRecorder>.
#[allow(non_snake_case)]
unsafe extern "system" fn ErrorRecorder_getNbErrors(this: *mut ErrorRecorder) -> i32 {
    this.as_ref().unwrap().rust_impl.nb_errors()
}
#[allow(non_snake_case)]
unsafe extern "system" fn ErrorRecorder_getErrorCode(
    this: *const ErrorRecorder,
    error_idx: i32,
) -> i32 {
    this.as_ref().unwrap().rust_impl.error_code(error_idx) as i32
}
#[allow(non_snake_case)]
unsafe extern "system" fn ErrorRecorder_getErrorDesc(
    this: *const ErrorRecorder,
    error_idx: i32,
    out_buf: *mut ::std::os::raw::c_char,
    out_buf_size: usize,
) {
    if out_buf.is_null() || out_buf_size == 0 {
        return;
    }
    let desc = this.as_ref().unwrap().rust_impl.error_desc(error_idx);
    let bytes = desc.to_bytes_with_nul();
    let copy_len = (bytes.len()).min(out_buf_size);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, copy_len);
    if copy_len < out_buf_size {
        *out_buf.add(copy_len) = 0;
    }
}
#[allow(non_snake_case)]
unsafe extern "system" fn ErrorRecorder_hasOverflowed(this: *mut ErrorRecorder) -> bool {
    this.as_ref().unwrap().rust_impl.has_overflowed()
}
#[allow(non_snake_case)]
unsafe extern "system" fn ErrorRecorder_clear(this: *mut ErrorRecorder) {
    this.as_mut().unwrap().rust_impl.clear();
}
#[allow(non_snake_case)]
unsafe extern "system" fn ErrorRecorder_reportError(
    this: *mut ErrorRecorder,
    val: i32,
    desc: *const ::std::os::raw::c_char,
) -> bool {
    let desc_str = CStr::from_ptr(desc).to_string_lossy();
    this.as_mut().unwrap().rust_impl.report_error(
        match val {
            0 => ErrorCode::kSUCCESS,
            1 => ErrorCode::kUNSPECIFIED_ERROR,
            2 => ErrorCode::kINTERNAL_ERROR,
            3 => ErrorCode::kINVALID_ARGUMENT,
            4 => ErrorCode::kINVALID_CONFIG,
            5 => ErrorCode::kFAILED_ALLOCATION,
            6 => ErrorCode::kFAILED_INITIALIZATION,
            7 => ErrorCode::kFAILED_EXECUTION,
            8 => ErrorCode::kFAILED_COMPUTATION,
            9 => ErrorCode::kINVALID_STATE,
            10 => ErrorCode::kUNSUPPORTED_STATE,
            _ => ErrorCode::kUNSPECIFIED_ERROR,
        },
        &desc_str,
    )
}
#[allow(non_snake_case)]
unsafe extern "system" fn ErrorRecorder_incRefCount(this: *mut ErrorRecorder) -> i32 {
    this.as_mut().unwrap().rust_impl.inc_ref_count()
}
#[allow(non_snake_case)]
unsafe extern "system" fn ErrorRecorder_decRefCount(this: *mut ErrorRecorder) -> i32 {
    this.as_mut().unwrap().rust_impl.dec_ref_count()
}

/// Bridges to [`trtx_sys::nvinfer1::IErrorRecorder`]; C++ [`nvinfer1::v_1_0::IErrorRecorder`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_error_recorder.html).
///
/// Construct with a [`RecordError`] implementation.
#[repr(C)]
pub struct ErrorRecorder {
    cpp_obj: UniquePtr<nvinfer1::IErrorRecorder>,
    rust_impl: Box<dyn RecordError>,
}

impl ErrorRecorder {
    pub fn new(inner: Box<dyn RecordError>) -> Result<Pin<Box<Self>>> {
        let mut rust_obj = Box::pin(ErrorRecorder {
            cpp_obj: UniquePtr::null(),
            rust_impl: inner,
        });
        unsafe {
            let cpp_obj = UniquePtr::from_raw(trtx_create_error_recorder(
                rust_obj.as_mut().get_unchecked_mut() as *mut ErrorRecorder
                    as *mut std::ffi::c_void,
                ErrorRecorder_getNbErrors as *mut std::ffi::c_void,
                ErrorRecorder_getErrorCode as *mut std::ffi::c_void,
                ErrorRecorder_getErrorDesc as *mut std::ffi::c_void,
                ErrorRecorder_hasOverflowed as *mut std::ffi::c_void,
                ErrorRecorder_clear as *mut std::ffi::c_void,
                ErrorRecorder_reportError as *mut std::ffi::c_void,
                ErrorRecorder_incRefCount as *mut std::ffi::c_void,
                ErrorRecorder_decRefCount as *mut std::ffi::c_void,
            ));
            if cpp_obj.is_null() {
                return Err(Error::Runtime(
                    "Failed to allocate object for IErrorRecorder subclass".to_string(),
                ));
            }
            rust_obj.cpp_obj = cpp_obj;
        }
        Ok(rust_obj)
    }

    pub fn as_trt_error_recorder(&self) -> *mut nvinfer1::IErrorRecorder {
        self.cpp_obj.as_mut_ptr()
    }
}

/// Implemented by [`ErrorRecorder`] for [`trtx_sys::nvinfer1::IErrorRecorder`]; C++ [`nvinfer1::v_1_0::IErrorRecorder`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_error_recorder.html).
pub trait RecordError: Send + Sync {
    fn nb_errors(&self) -> i32;
    fn error_code(&self, error_idx: i32) -> ErrorCode;
    fn error_desc(&self, error_idx: i32) -> &CStr;
    fn has_overflowed(&self) -> bool;
    fn clear(&self);
    fn report_error(&self, val: ErrorCode, desc: &str) -> bool;
    fn inc_ref_count(&self) -> i32;
    fn dec_ref_count(&self) -> i32;
}

#[allow(non_snake_case)]
unsafe extern "system" fn DebugListener_processDebugTensor(
    this: *const std::ffi::c_void,
    addr: *const std::ffi::c_void,
    location: nvinfer1::TensorLocation,
    type_: nvinfer1::DataType,
    shape: *const Dims64,
    name: *const std::ffi::c_char,
    stream: *mut std::ffi::c_void,
) -> bool {
    let this = this as *const DebugListener;
    let name = (!name.is_null()).then(|| CStr::from_ptr(name));
    let name = name.map(|s| s.to_string_lossy());
    this.as_ref()
        .unwrap()
        .rust_impl
        .process_debug_tensor(
            addr,
            location.into(),
            type_.into(),
            shape.as_ref().unwrap(),
            name.as_deref(),
            stream,
        )
        .is_ok()
}

/// Bridges to [`trtx_sys::nvinfer1::IDebugListener`]; C++ [`nvinfer1::v_1_0::IDebugListener`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_debug_listener.html).
#[repr(C)]
pub struct DebugListener {
    cpp_obj: *mut nvinfer1::IDebugListener,
    rust_impl: Box<dyn ProcessDebugTensor>,
}

pub type ProcessDebugTensorResult = std::result::Result<(), ()>;

impl DebugListener {
    pub fn new(inner: Box<dyn ProcessDebugTensor>) -> Result<Pin<Box<Self>>> {
        let mut rust_obj = Box::pin(Self {
            cpp_obj: null_mut(),
            rust_impl: inner,
        });
        unsafe {
            let cpp_obj = trtx_create_debug_listener(
                rust_obj.as_mut().get_unchecked_mut() as *mut DebugListener
                    as *mut std::ffi::c_void,
                DebugListener_processDebugTensor,
            );
            if cpp_obj.is_null() {
                return Err(Error::Runtime(
                    "Failed to allocate object for IDebugListener subclass".to_string(),
                ));
            }
            rust_obj.cpp_obj = cpp_obj;
        }
        Ok(rust_obj)
    }

    pub fn as_raw(&self) -> *mut nvinfer1::IDebugListener {
        self.cpp_obj
    }
}

/// Implemented by [`DebugListener`] for [`trtx_sys::nvinfer1::IDebugListener`] (`processDebugTensor`); C++ [`nvinfer1::v_1_0::IDebugListener`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1v__1__0_1_1_i_debug_listener.html).
pub trait ProcessDebugTensor: Send + Sync {
    /// # Safety
    ///
    /// User needs to ensure memory safety for CUDA pointers and ensure correct lifetimes for CUDA
    /// objects
    #[allow(clippy::result_unit_err)]
    unsafe fn process_debug_tensor(
        &self,
        addr: *const std::ffi::c_void,
        location: TensorLocation,
        type_: DataType,
        shape: &Dims64,
        name: Option<&str>,
        stream: *mut std::ffi::c_void,
    ) -> ProcessDebugTensorResult;
}

//#[subclass]
//#[derive(Default)]
//pub struct StreamReaderV2 {
//inner: Option<Box<dyn ReadStreamV2>>,
//}

//impl StreamReaderV2 {
//pub fn new(inner: Box<dyn ReadStreamV2>) -> Rc<RefCell<Self>> {
//let rtn = Self::default_rust_owned();
//rtn.borrow_mut().inner = Some(inner);
//rtn
//}
//}

//impl nvinfer1::IStreamReaderV2_methods for StreamReaderV2 {
//unsafe fn read(
//&mut self,
//destination: *mut autocxx::c_void,
//nbBytes: i64,
//stream: *mut crate::ffi::CUstream_st,
//) -> i64 {
//self.inner
//.as_mut()
//.unwrap()
//.read(destination, nbBytes, stream)
//}
//fn seek(&mut self, offset: i64, where_: nvinfer1::SeekPosition) -> bool {
//self.inner.as_mut().unwrap().seek(offset, where_.into())
//}
//}

//pub trait ReadStreamV2: Send + Sync {
//unsafe fn read(
//&mut self,
//destination: *mut autocxx::c_void,
//nbBytes: i64,
//stream: *mut crate::ffi::CUstream_st,
//) -> i64;
//fn seek(&mut self, offset: i64, where_: crate::SeekPosition) -> bool;
//}
