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
    fn phase_start(&mut self, phase_name: &str, parent_phase: Option<&str>, num_steps: i32);
    /// See [nvinfer::IProgressMonitor::stepComplete]. Return whether to continue building or cancel
    fn step_complete(&mut self, phase_name: &str, step: i32) -> std::ops::ControlFlow<()>;
    /// See [nvinfer::IProgressMonitor::phaseFinish]
    fn phase_finish(&mut self, phase_name: &str);
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
        unsafe { trtx_destroy_progress_monitor_subclass(self.cpp_obj) }
    }
}

#[subclass]
#[derive(Default)]
pub struct GpuAllocator {
    inner: Option<Box<dyn AllocateGpu>>,
}

impl GpuAllocator {
    pub fn new(inner: Box<dyn AllocateGpu>) -> Rc<RefCell<Self>> {
        let rtn = Self::default_rust_owned();
        rtn.borrow_mut().inner = Some(inner);
        rtn
    }
}

impl nvinfer1::IGpuAllocator_methods for GpuAllocator {
    fn allocate(&mut self, size: u64, alignment: u64, flags: u32) -> *mut autocxx::c_void {
        unsafe {
            self.inner
                .as_mut()
                .unwrap()
                .allocate_async(size, alignment, flags, null_mut())
        }
    }
    unsafe fn allocateAsync(
        &mut self,
        size: u64,
        alignment: u64,
        flags: u32,
        cuda_stream: *mut crate::ffi::CUstream_st,
    ) -> *mut autocxx::c_void {
        unsafe {
            self.inner
                .as_mut()
                .unwrap()
                .allocate_async(size, alignment, flags, cuda_stream)
        }
    }
    unsafe fn reallocate(
        &mut self,
        memory: *mut autocxx::c_void,
        alignment: u64,
        new_size: u64,
    ) -> *mut autocxx::c_void {
        unsafe {
            self.inner
                .as_mut()
                .unwrap()
                .reallocate(memory, alignment, new_size)
        }
    }

    unsafe fn deallocate(&mut self, data: *mut autocxx::c_void) -> bool {
        unsafe {
            self.inner
                .as_mut()
                .unwrap()
                .deallocate_async(data, null_mut())
        }
    }
    unsafe fn deallocateAsync(
        &mut self,
        data: *mut autocxx::c_void,
        cuda_stream: *mut crate::ffi::CUstream_st,
    ) -> bool {
        self.inner
            .as_mut()
            .unwrap()
            .deallocate_async(data, cuda_stream)
    }
}

pub trait AllocateGpu: Send + Sync {
    // we omit the following deprecated methods
    //fn allocate(&mut self, size: u64, alignment: u64, flags: u32) -> *mut autocxx::c_void;
    //unsafe fn deallocate(&mut self, data: *mut autocxx::c_void) -> bool;
    unsafe fn allocate_async(
        &mut self,
        size: u64,
        alignment: u64,
        flags: u32,
        cuda_stream: *mut crate::ffi::CUstream_st,
    ) -> *mut autocxx::c_void;
    unsafe fn reallocate(
        &mut self,
        memory: *mut autocxx::c_void,
        alignment: u64,
        new_size: u64,
    ) -> *mut autocxx::c_void;
    unsafe fn deallocate_async(
        &mut self,
        data: *mut autocxx::c_void,
        cuda_stream: *mut crate::ffi::CUstream_st,
    ) -> bool;
}

#[subclass]
#[derive(Default)]
pub struct ErrorRecorder {
    inner: Option<Box<dyn RecordError>>,
}

impl ErrorRecorder {
    pub fn new(inner: Box<dyn RecordError>) -> Rc<RefCell<Self>> {
        let rtn = Self::default_rust_owned();
        rtn.borrow_mut().inner = Some(inner);
        rtn
    }
}

impl nvinfer1::IErrorRecorder_methods for ErrorRecorder {
    fn getNbErrors(&self) -> i32 {
        self.inner.as_ref().unwrap().nb_errors()
    }
    fn getErrorCode(&self, errorIdx: i32) -> i32 {
        self.inner.as_ref().unwrap().error_code(errorIdx) as i32
    }
    fn getErrorDesc(&self, errorIdx: i32) -> *const ::std::os::raw::c_char {
        self.inner.as_ref().unwrap().error_desc(errorIdx).as_ptr()
    }
    fn hasOverflowed(&self) -> bool {
        self.inner.as_ref().unwrap().has_overflowed()
    }
    fn clear(&mut self) {
        self.inner.as_mut().unwrap().clear()
    }
    unsafe fn reportError(&mut self, val: i32, desc: *const ::std::os::raw::c_char) -> bool {
        let desc = CStr::from_ptr(desc);
        self.inner.as_mut().unwrap().report_error(
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
            &desc.to_string_lossy(),
        )
    }
    fn incRefCount(&mut self) -> i32 {
        self.inner.as_mut().unwrap().inc_ref_count()
    }
    fn decRefCount(&mut self) -> i32 {
        self.inner.as_mut().unwrap().dec_ref_count()
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
