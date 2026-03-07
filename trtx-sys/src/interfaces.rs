use crate::{enums::ErrorCode, ffi, DataType, Dims64, Severity, TensorLocation};
use ::autocxx::subclass::*;
use std::{
    cell::RefCell,
    ffi::{c_char, CStr},
    pin::Pin,
    ptr::null_mut,
    rc::Rc,
};

use crate::nvinfer1;

/// Rust version of the [nvinfer1::IProgressMonitor]
///
/// Put into a [ProgressMonitor] to subclass [nvinfer1::IProgressMonitor]
pub trait HandleProgress: Send + Sync {
    /// See [nvinfer::IProgressMonitor::phaseStart]
    fn phase_start(&self, phase_name: &str, parent_phase: Option<&str>, num_steps: i32);
    /// See [nvinfer::IProgressMonitor::stepComplete]. Return whether to continue building or cancel
    fn step_complete(&self, phase_name: &str, step: i32) -> std::ops::ControlFlow<()>;
    /// See [nvinfer::IProgressMonitor::phaseFinish]
    fn phase_finish(&self, phase_name: &str);
}

unsafe extern "system" fn ProgressMonitor_phaseStart(
    this: *mut ProgressMonitor,
    phaseName: *const ::std::os::raw::c_char,
    parentPhase: *const ::std::os::raw::c_char,
    nbSteps: i32,
) {
    let this = this as *mut ProgressMonitor;
    let phase_name = CStr::from_ptr(phaseName);
    let parent_phase = if parentPhase.is_null() {
        None
    } else {
        Some(CStr::from_ptr(phaseName).to_string_lossy())
    };
    this.as_mut().unwrap().rust_impl.phase_start(
        &phase_name.to_string_lossy(),
        parent_phase.as_ref().map(|v| v.as_ref()),
        nbSteps,
    );
}
unsafe extern "system" fn ProgressMonitor_stepComplete(
    this: *mut ProgressMonitor,
    phaseName: *const ::std::os::raw::c_char,
    step: i32,
) -> bool {
    let phase_name = CStr::from_ptr(phaseName);
    this.as_mut()
        .unwrap()
        .rust_impl
        .step_complete(&phase_name.to_string_lossy(), step)
        .is_continue()
}
unsafe extern "system" fn ProgressMonitor_phaseFinish(
    this: *mut ProgressMonitor,
    phaseName: *const ::std::os::raw::c_char,
) {
    let phase_name = CStr::from_ptr(phaseName);
    this.as_mut()
        .unwrap()
        .rust_impl
        .phase_finish(&phase_name.to_string_lossy());
}

extern "C" {
    fn trtx_create_progress_monitor_subclass(
        rust_impl: *mut std::ffi::c_void,
        phaseStart: *mut std::ffi::c_void,
        stepComplete: *mut std::ffi::c_void,
        phaseFinish: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;
    fn trtx_destroy_progress_monitor_subclass(cpp_obj: *mut std::ffi::c_void);
}

///
/// Subclasses [nvinfer1::IProgressMonitor]
///
/// Construct a object with a dyn [HandleProgress] to implement
/// [nvinfer1::IProgressMonitor] from Rust
#[repr(C)]
pub struct ProgressMonitor {
    cpp_obj: *mut std::ffi::c_void,
    rust_impl: Box<dyn HandleProgress>,
}

impl ProgressMonitor {
    pub fn new(inner: Box<dyn HandleProgress>) -> Pin<Box<ProgressMonitor>> {
        let mut rust_obj = Box::pin(ProgressMonitor {
            cpp_obj: null_mut(),
            rust_impl: inner,
        });

        unsafe {
            let cpp_obj = trtx_create_progress_monitor_subclass(
                rust_obj.as_mut().get_unchecked_mut() as *mut ProgressMonitor
                    as *mut std::ffi::c_void,
                ProgressMonitor_phaseStart as *mut std::ffi::c_void,
                ProgressMonitor_stepComplete as *mut std::ffi::c_void,
                ProgressMonitor_phaseFinish as *mut std::ffi::c_void,
            );
            rust_obj.as_mut().get_unchecked_mut().cpp_obj = cpp_obj;
        }
        rust_obj
    }
    pub fn as_trt_progress_monitor(&self) -> *mut nvinfer1::IProgressMonitor {
        self.cpp_obj as *mut nvinfer1::IProgressMonitor
    }
}

impl Drop for ProgressMonitor {
    fn drop(&mut self) {
        if !self.cpp_obj.is_null() {
            unsafe { trtx_destroy_progress_monitor_subclass(self.cpp_obj) }
        }
    }
}

/// C callbacks for GpuAllocatorSubclass (bridge to Rust). `this` is *mut RefCell<GpuAllocator>.
unsafe extern "system" fn GpuAllocator_allocateAsync(
    this: *const GpuAllocator,
    size: u64,
    alignment: u64,
    flags: u32,
    cuda_stream: *mut crate::ffi::CUstream_st,
) -> *mut std::ffi::c_void {
    this.as_ref()
        .unwrap()
        .rust_impl
        .allocate_async(size, alignment, flags, cuda_stream)
}
unsafe extern "system" fn GpuAllocator_reallocate(
    this: *const GpuAllocator,
    memory: *mut std::ffi::c_void,
    alignment: u64,
    new_size: u64,
) -> *mut std::ffi::c_void {
    this.as_ref()
        .unwrap()
        .rust_impl
        .reallocate(memory, alignment, new_size)
}
unsafe extern "system" fn GpuAllocator_deallocateAsync(
    this: *mut std::cell::RefCell<GpuAllocator>,
    memory: *mut std::ffi::c_void,
    cuda_stream: *mut crate::ffi::CUstream_st,
) -> bool {
    this.as_mut()
        .unwrap()
        .borrow_mut()
        .rust_impl
        .deallocate_async(memory, cuda_stream)
}

extern "C" {
    fn trtx_create_gpu_allocator_subclass(
        rust_impl: *mut std::ffi::c_void,
        allocateAsync: *mut std::ffi::c_void,
        reallocate: *mut std::ffi::c_void,
        deallocateAsync: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;
    fn trtx_destroy_gpu_allocator_subclass(cpp_obj: *mut std::ffi::c_void);
}

///
/// Subclasses [nvinfer1::IGpuAllocator] via C++ bridge.
///
/// Construct with an [AllocateGpu] to implement [nvinfer1::IGpuAllocator] from Rust.
#[repr(C)]
pub struct GpuAllocator {
    cpp_obj: *mut std::ffi::c_void,
    rust_impl: Box<dyn AllocateGpu>,
}

impl GpuAllocator {
    pub fn new(inner: Box<dyn AllocateGpu>) -> Rc<RefCell<Self>> {
        let rust_obj = Rc::new(RefCell::new(GpuAllocator {
            cpp_obj: null_mut(),
            rust_impl: inner,
        }));
        unsafe {
            let cpp_obj = trtx_create_gpu_allocator_subclass(
                rust_obj.as_ptr() as *mut std::ffi::c_void,
                GpuAllocator_allocateAsync as *mut std::ffi::c_void,
                GpuAllocator_reallocate as *mut std::ffi::c_void,
                GpuAllocator_deallocateAsync as *mut std::ffi::c_void,
            );
            rust_obj.borrow_mut().cpp_obj = cpp_obj;
        }
        rust_obj
    }

    pub fn as_trt_gpu_allocator(&self) -> *mut nvinfer1::IGpuAllocator {
        self.cpp_obj as *mut nvinfer1::IGpuAllocator
    }
}

impl Drop for GpuAllocator {
    fn drop(&mut self) {
        if !self.cpp_obj.is_null() {
            unsafe { trtx_destroy_gpu_allocator_subclass(self.cpp_obj) }
            self.cpp_obj = null_mut();
        }
    }
}

pub trait AllocateGpu: Send + Sync {
    // we omit the following deprecated methods
    //fn allocate(&mut self, size: u64, alignment: u64, flags: u32) -> *mut autocxx::c_void;
    //unsafe fn deallocate(&mut self, data: *mut autocxx::c_void) -> bool;
    unsafe fn allocate_async(
        &self,
        size: u64,
        alignment: u64,
        flags: u32,
        cuda_stream: *mut crate::ffi::CUstream_st,
    ) -> *mut std::ffi::c_void;
    unsafe fn reallocate(
        &self,
        memory: *mut std::ffi::c_void,
        alignment: u64,
        new_size: u64,
    ) -> *mut std::ffi::c_void;
    unsafe fn deallocate_async(
        &self,
        data: *mut std::ffi::c_void,
        cuda_stream: *mut crate::ffi::CUstream_st,
    ) -> bool;
}

/// C callbacks for ErrorRecorderSubclass (bridge to Rust). `this` is *mut RefCell<ErrorRecorder>.
unsafe extern "system" fn ErrorRecorder_getNbErrors(this: *mut ErrorRecorder) -> i32 {
    this.as_ref().unwrap().rust_impl.nb_errors()
}
unsafe extern "system" fn ErrorRecorder_getErrorCode(
    this: *const ErrorRecorder,
    error_idx: i32,
) -> i32 {
    this.as_ref().unwrap().rust_impl.error_code(error_idx) as i32
}
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
unsafe extern "system" fn ErrorRecorder_hasOverflowed(
    this: *mut std::cell::RefCell<ErrorRecorder>,
) -> bool {
    this.as_ref().unwrap().borrow().rust_impl.has_overflowed()
}
unsafe extern "system" fn ErrorRecorder_clear(this: *mut std::cell::RefCell<ErrorRecorder>) {
    this.as_mut().unwrap().borrow_mut().rust_impl.clear();
}
unsafe extern "system" fn ErrorRecorder_reportError(
    this: *mut std::cell::RefCell<ErrorRecorder>,
    val: i32,
    desc: *const ::std::os::raw::c_char,
) -> bool {
    let desc_str = CStr::from_ptr(desc).to_string_lossy();
    this.as_mut().unwrap().borrow_mut().rust_impl.report_error(
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
unsafe extern "system" fn ErrorRecorder_incRefCount(
    this: *mut std::cell::RefCell<ErrorRecorder>,
) -> i32 {
    this.as_mut()
        .unwrap()
        .borrow_mut()
        .rust_impl
        .inc_ref_count()
}
unsafe extern "system" fn ErrorRecorder_decRefCount(
    this: *mut std::cell::RefCell<ErrorRecorder>,
) -> i32 {
    this.as_mut()
        .unwrap()
        .borrow_mut()
        .rust_impl
        .dec_ref_count()
}

extern "C" {
    fn trtx_create_error_recorder_subclass(
        rust_impl: *mut std::ffi::c_void,
        getNbErrors: *mut std::ffi::c_void,
        getErrorCode: *mut std::ffi::c_void,
        getErrorDesc: *mut std::ffi::c_void,
        hasOverflowed: *mut std::ffi::c_void,
        clear: *mut std::ffi::c_void,
        reportError: *mut std::ffi::c_void,
        incRefCount: *mut std::ffi::c_void,
        decRefCount: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;
    fn trtx_destroy_error_recorder_subclass(cpp_obj: *mut std::ffi::c_void);
}

///
/// Subclasses [nvinfer1::IErrorRecorder] via C++ bridge.
///
/// Construct with a [RecordError] to implement [nvinfer1::IErrorRecorder] from Rust.
#[repr(C)]
pub struct ErrorRecorder {
    cpp_obj: *mut std::ffi::c_void,
    rust_impl: Box<dyn RecordError>,
}

impl ErrorRecorder {
    pub fn new(inner: Box<dyn RecordError>) -> Rc<RefCell<Self>> {
        let rust_obj = Rc::new(RefCell::new(ErrorRecorder {
            cpp_obj: null_mut(),
            rust_impl: inner,
        }));
        unsafe {
            let cpp_obj = trtx_create_error_recorder_subclass(
                rust_obj.as_ptr() as *mut std::ffi::c_void,
                ErrorRecorder_getNbErrors as *mut std::ffi::c_void,
                ErrorRecorder_getErrorCode as *mut std::ffi::c_void,
                ErrorRecorder_getErrorDesc as *mut std::ffi::c_void,
                ErrorRecorder_hasOverflowed as *mut std::ffi::c_void,
                ErrorRecorder_clear as *mut std::ffi::c_void,
                ErrorRecorder_reportError as *mut std::ffi::c_void,
                ErrorRecorder_incRefCount as *mut std::ffi::c_void,
                ErrorRecorder_decRefCount as *mut std::ffi::c_void,
            );
            rust_obj.borrow_mut().cpp_obj = cpp_obj;
        }
        rust_obj
    }

    pub fn as_trt_error_recorder(&self) -> *mut nvinfer1::IErrorRecorder {
        self.cpp_obj as *mut nvinfer1::IErrorRecorder
    }
}

impl Drop for ErrorRecorder {
    fn drop(&mut self) {
        if !self.cpp_obj.is_null() {
            unsafe { trtx_destroy_error_recorder_subclass(self.cpp_obj) }
            self.cpp_obj = null_mut();
        }
    }
}

pub trait RecordError: Send + Sync {
    fn nb_errors(&self) -> i32;
    fn error_code(&self, error_idx: i32) -> ErrorCode;
    fn error_desc(&self, error_idx: i32) -> &CStr;
    fn has_overflowed(&self) -> bool;
    fn clear(&mut self);
    unsafe fn report_error(&mut self, val: ErrorCode, desc: &str) -> bool;
    fn inc_ref_count(&mut self) -> i32;
    fn dec_ref_count(&mut self) -> i32;
}

#[subclass]
#[derive(Default)]
pub struct DebugListener {
    inner: Option<Box<dyn ProcessDebugTensor>>,
}

impl DebugListener {
    pub fn new(inner: Box<dyn ProcessDebugTensor>) -> Rc<RefCell<Self>> {
        let rtn = Self::default_rust_owned();
        rtn.borrow_mut().inner = Some(inner);
        rtn
    }
}

impl nvinfer1::IDebugListener_methods for DebugListener {
    unsafe fn processDebugTensor(
        &mut self,
        addr: *const autocxx::c_void,
        location: nvinfer1::TensorLocation,
        type_: nvinfer1::DataType,
        shape: &Dims64,
        name: *const c_char,
        stream: *mut ffi::CUstream_st,
    ) -> bool {
        let name = CStr::from_ptr(name);
        self.inner.as_mut().unwrap().process_debug_tensor(
            addr,
            location.into(),
            type_.into(),
            shape,
            &name.to_string_lossy(),
            stream,
        )
    }
}

pub trait ProcessDebugTensor: Send + Sync {
    unsafe fn process_debug_tensor(
        &mut self,
        addr: *const autocxx::c_void,
        location: TensorLocation,
        type_: DataType,
        shape: &Dims64,
        name: &str,
        stream: *mut ffi::CUstream_st,
    ) -> bool;
}

#[subclass]
#[derive(Default)]
pub struct Logger {
    inner: Option<Box<dyn HandleLog>>,
}

impl Logger {
    pub fn new(inner: Box<dyn HandleLog>) -> Rc<RefCell<Self>> {
        let rtn = Self::default_rust_owned();
        rtn.borrow_mut().inner = Some(inner);
        rtn
    }
}

impl nvinfer1::ILogger_methods for Logger {
    unsafe fn log(&mut self, severity: i32, message: *const i8) {
        let message = CStr::from_ptr(message);
        self.inner.as_mut().unwrap().log(
            match severity {
                0 => Severity::kINTERNAL_ERROR,
                1 => Severity::kERROR,
                2 => Severity::kWARNING,
                3 => Severity::kINFO,
                4 | _ => Severity::kVERBOSE,
            },
            &message.to_string_lossy(),
        )
    }
}

pub trait HandleLog: Send + Sync {
    unsafe fn log(&mut self, severity: Severity, message: &str);
}

#[subclass]
#[derive(Default)]
pub struct StreamReaderV2 {
    inner: Option<Box<dyn ReadStreamV2>>,
}

impl StreamReaderV2 {
    pub fn new(inner: Box<dyn ReadStreamV2>) -> Rc<RefCell<Self>> {
        let rtn = Self::default_rust_owned();
        rtn.borrow_mut().inner = Some(inner);
        rtn
    }
}

impl nvinfer1::IStreamReaderV2_methods for StreamReaderV2 {
    unsafe fn read(
        &mut self,
        destination: *mut autocxx::c_void,
        nbBytes: i64,
        stream: *mut crate::ffi::CUstream_st,
    ) -> i64 {
        self.inner
            .as_mut()
            .unwrap()
            .read(destination, nbBytes, stream)
    }
    fn seek(&mut self, offset: i64, where_: nvinfer1::SeekPosition) -> bool {
        self.inner.as_mut().unwrap().seek(offset, where_.into())
    }
}

pub trait ReadStreamV2: Send + Sync {
    unsafe fn read(
        &mut self,
        destination: *mut autocxx::c_void,
        nbBytes: i64,
        stream: *mut crate::ffi::CUstream_st,
    ) -> i64;
    fn seek(&mut self, offset: i64, where_: crate::SeekPosition) -> bool;
}
