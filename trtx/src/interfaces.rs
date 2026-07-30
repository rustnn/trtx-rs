//! Rust implementations of TensorRT callback / allocator interfaces (bridged to C++).
//!
//! Versioned runtime interfaces live under `nvinfer1::v_1_0` in C++; see the
//! [annotated class list](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/annotated.html).

use crate::{Error, Result};
use cxx::UniquePtr;
use std::ptr::null_mut;
use std::{ffi::CStr, marker::PhantomPinned, pin::Pin};
use trtx_sys::{
    nvinfer1, trtx_create_debug_listener, trtx_create_error_recorder, trtx_create_gpu_allocator,
    trtx_create_profiler, trtx_create_progress_monitor, trtx_create_stream_reader_v2,
    trtx_destroy_profiler, trtx_destroy_stream_reader_v2,
};
use trtx_sys::{DataType, Dims64, ErrorCode, SeekPosition, TensorLocation};

/// Rust trait implemented by [`ProgressMonitor`] for [`trtx_sys::nvinfer1::IProgressMonitor`]; C++ [`nvinfer1::v_1_0::IProgressMonitor`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_progress_monitor.html).
///
/// Use with [`crate::BuilderConfig::set_progress_monitor`].
pub trait MonitorProgress: Send + Sync {
    /// See [`trtx_sys::nvinfer1::IProgressMonitor`] / C++ [`phaseStart`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_progress_monitor.html).
    fn phase_start(&self, phase_name: &str, parent_phase: Option<&str>, num_steps: i32);
    /// See [`trtx_sys::nvinfer1::IProgressMonitor`] / C++ [`stepComplete`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_progress_monitor.html). Return whether to continue building or cancel.
    fn step_complete(&self, phase_name: &str, step: i32) -> std::ops::ControlFlow<()>;
    /// See [`trtx_sys::nvinfer1::IProgressMonitor`] / C++ [`phaseFinish`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_progress_monitor.html).
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

/// Bridges to [`trtx_sys::nvinfer1::IProgressMonitor`]; C++ [`nvinfer1::v_1_0::IProgressMonitor`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_progress_monitor.html).
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

/// Reads a serialized TensorRT engine plan for [`StreamReaderV2`].
///
/// TensorRT may request reads into either host or CUDA device memory. Implementations commonly
/// use `cudaPointerGetAttributes` to determine the destination's memory location.
pub trait ReadStreamV2: Send + Sync {
    /// Reads up to `byte_count` bytes into `destination`, using `stream` for any asynchronous
    /// CUDA work.
    ///
    /// Return the number of bytes read, `0` at end of stream, or a negative value for an
    /// unrecoverable error.
    ///
    /// # Safety
    ///
    /// TensorRT supplies `destination` and `stream`. Implementations must treat them according to
    /// CUDA's pointer and stream rules, write at most `byte_count` bytes, and ensure any
    /// asynchronous work is enqueued on `stream`.
    unsafe fn read(
        &mut self,
        destination: *mut std::ffi::c_void,
        byte_count: i64,
        stream: *mut std::ffi::c_void,
    ) -> i64;

    /// Moves the reader by `offset` bytes relative to `position`.
    fn seek(&mut self, offset: i64, position: SeekPosition) -> bool;
}

#[allow(non_snake_case)]
unsafe extern "system" fn StreamReaderV2_read(
    this: *mut std::ffi::c_void,
    destination: *mut std::ffi::c_void,
    byte_count: i64,
    stream: *mut std::ffi::c_void,
) -> i64 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let this = unsafe { &mut *this.cast::<StreamReaderV2>() };
        unsafe { this.rust_impl.read(destination, byte_count, stream) }
    }))
    .unwrap_or(-1)
}

#[allow(non_snake_case)]
unsafe extern "system" fn StreamReaderV2_seek(
    this: *mut std::ffi::c_void,
    offset: i64,
    position: i32,
) -> bool {
    let position = match position {
        0 => SeekPosition::kSET,
        1 => SeekPosition::kCUR,
        2 => SeekPosition::kEND,
        _ => return false,
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let this = unsafe { &mut *this.cast::<StreamReaderV2>() };
        this.rust_impl.seek(offset, position)
    }))
    .unwrap_or(false)
}

/// Rust implementation of TensorRT's `IStreamReaderV2` interface.
///
/// The returned object is pinned because its C++ bridge retains a pointer to the Rust wrapper.
/// Use it with `CudaEngine::load_weights_async` or other stream-reader APIs.
#[repr(C)]
pub struct StreamReaderV2 {
    cpp_obj: *mut nvinfer1::IStreamReaderV2,
    rust_impl: Box<dyn ReadStreamV2>,
    _pin: PhantomPinned,
}

impl std::fmt::Debug for StreamReaderV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamReaderV2")
            .field("inner", &(self.cpp_obj as usize))
            .finish_non_exhaustive()
    }
}

impl StreamReaderV2 {
    /// Creates a pinned TensorRT stream reader backed by `inner`.
    pub fn new(inner: Box<dyn ReadStreamV2>) -> Result<Pin<Box<Self>>> {
        let mut rust_obj = Box::pin(Self {
            cpp_obj: null_mut(),
            rust_impl: inner,
            _pin: PhantomPinned,
        });
        let cpp_obj = unsafe {
            trtx_create_stream_reader_v2(
                rust_obj.as_mut().get_unchecked_mut() as *mut Self as *mut std::ffi::c_void,
                StreamReaderV2_read,
                StreamReaderV2_seek,
            )
        };
        if cpp_obj.is_null() {
            return Err(Error::Runtime(
                "Failed to allocate object for IStreamReaderV2 subclass".to_string(),
            ));
        }
        unsafe {
            rust_obj.as_mut().get_unchecked_mut().cpp_obj = cpp_obj;
        }
        Ok(rust_obj)
    }

    #[allow(dead_code)] // Also used by stream APIs that are not wrapped yet.
    pub(crate) fn as_trt_stream_reader(
        self: Pin<&mut Self>,
    ) -> Pin<&mut nvinfer1::IStreamReaderV2> {
        let cpp_obj = self.as_ref().get_ref().cpp_obj;
        // SAFETY: `new` establishes that this pointer is non-null and uniquely owned by `self`.
        // The returned borrow cannot outlive the exclusive pinned borrow of the Rust wrapper.
        unsafe { Pin::new_unchecked(&mut *cpp_obj) }
    }
}

impl Drop for StreamReaderV2 {
    fn drop(&mut self) {
        if !self.cpp_obj.is_null() {
            unsafe {
                trtx_destroy_stream_reader_v2(self.cpp_obj);
            }
            self.cpp_obj = null_mut();
        }
    }
}

/// C callbacks for GpuAllocatorSubclass (bridge to Rust). `this` is `*mut RefCell<GpuAllocator>`.
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

/// Bridges to [`trtx_sys::nvinfer1::IGpuAllocator`]; C++ [`nvinfer1::v_1_0::IGpuAllocator`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_gpu_allocator.html).
///
/// Construct with an [`AllocateGpu`] implementation.
#[repr(C)]
pub struct GpuAllocator {
    cpp_obj: UniquePtr<nvinfer1::IGpuAllocator>,
    rust_impl: Box<dyn AllocateGpu>,
}

impl std::fmt::Debug for GpuAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuAllocator")
            .field("inner", &format!("{:x}", self.cpp_obj.as_ptr() as usize))
            .finish_non_exhaustive()
    }
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

/// Implemented by [`GpuAllocator`] for [`trtx_sys::nvinfer1::IGpuAllocator`]; C++ [`nvinfer1::v_1_0::IGpuAllocator`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_gpu_allocator.html).
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

/// C callbacks for ErrorRecorderSubclass (bridge to Rust). `this` is `*mut RefCell<ErrorRecorder>`.
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

/// Bridges to [`trtx_sys::nvinfer1::IErrorRecorder`]; C++ [`nvinfer1::v_1_0::IErrorRecorder`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_error_recorder.html).
///
/// Construct with a [`RecordError`] implementation.
#[repr(C)]
pub struct ErrorRecorder {
    cpp_obj: UniquePtr<nvinfer1::IErrorRecorder>,
    rust_impl: Box<dyn RecordError>,
}

impl std::fmt::Debug for ErrorRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorRecorder")
            .field("inner", &format!("{:x}", self.cpp_obj.as_ptr() as usize))
            .finish_non_exhaustive()
    }
}

/// # Safety
///
/// Send and Sync since container only initialized in `new` and
/// IErrorRecorder requires subclasses to be thread safe (we ensure this with RecordError: Send + Sync)
unsafe impl Send for ErrorRecorder {}
unsafe impl Sync for ErrorRecorder {}

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

/// Implemented by [`ErrorRecorder`] for [`trtx_sys::nvinfer1::IErrorRecorder`]; C++ [`nvinfer1::v_1_0::IErrorRecorder`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_error_recorder.html).
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

/// Bridges to [`trtx_sys::nvinfer1::IDebugListener`]; C++ [`nvinfer1::v_1_0::IDebugListener`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_debug_listener.html).
#[repr(C)]
pub struct DebugListener {
    cpp_obj: *mut nvinfer1::IDebugListener,
    rust_impl: Box<dyn ProcessDebugTensor>,
}

pub type ProcessDebugTensorResult = std::result::Result<(), ()>;

impl std::fmt::Debug for DebugListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DebugListener")
            .field("inner", &format!("{:x}", self.cpp_obj as usize))
            .finish_non_exhaustive()
    }
}

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

/// Implemented by [`DebugListener`] for [`trtx_sys::nvinfer1::IDebugListener`] (`processDebugTensor`); C++ [`nvinfer1::v_1_0::IDebugListener`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_debug_listener.html).
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

#[allow(non_snake_case)]
unsafe extern "system" fn Profiler_reportLayerTime(
    this: *mut std::ffi::c_void,
    layerName: *const ::std::os::raw::c_char,
    ms: f32,
) {
    let this = this as *mut Profiler;
    let name = if layerName.is_null() {
        std::borrow::Cow::Borrowed("")
    } else {
        CStr::from_ptr(layerName).to_string_lossy()
    };
    this.as_ref()
        .unwrap()
        .rust_impl
        .report_layer_time(name.as_ref(), ms);
}

/// Bridges to [`trtx_sys::nvinfer1::IProfiler`]; C++ [`nvinfer1::v_1_0::IProfiler`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_profiler.html).
#[repr(C)]
pub struct Profiler {
    cpp_obj: *mut nvinfer1::IProfiler,
    rust_impl: Box<dyn ReportLayerTime>,
}

impl std::fmt::Debug for Profiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Profiler")
            .field("inner", &format!("{:x}", self.cpp_obj as usize))
            .finish_non_exhaustive()
    }
}

impl Profiler {
    pub fn new(inner: Box<dyn ReportLayerTime>) -> Result<Pin<Box<Self>>> {
        let mut rust_obj = Box::pin(Self {
            cpp_obj: null_mut(),
            rust_impl: inner,
        });
        unsafe {
            let cpp_obj = trtx_create_profiler(
                rust_obj.as_mut().get_unchecked_mut() as *mut Profiler as *mut std::ffi::c_void,
                Profiler_reportLayerTime,
            );
            if cpp_obj.is_null() {
                return Err(Error::Runtime(
                    "Failed to allocate object for IProfiler subclass".to_string(),
                ));
            }
            rust_obj.cpp_obj = cpp_obj;
        }
        Ok(rust_obj)
    }

    pub fn as_raw(&self) -> *mut nvinfer1::IProfiler {
        self.cpp_obj
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        if !self.cpp_obj.is_null() {
            unsafe {
                trtx_destroy_profiler(self.cpp_obj);
            }
            self.cpp_obj = null_mut();
        }
    }
}

/// Implemented by [`Profiler`] for [`trtx_sys::nvinfer1::IProfiler`] (`reportLayerTime`); C++ [`nvinfer1::v_1_0::IProfiler`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1v__1__0_1_1_i_profiler.html).
pub trait ReportLayerTime: Send + Sync {
    /// Layer name from the network (or a decimal layer index if the engine was built with profiling
    /// verbosity [`crate::ProfilingVerbosity::kNONE`]). `ms` is execution time for that layer in
    /// milliseconds.
    fn report_layer_time(&self, layer_name: &str, ms: f32);
}

#[cfg(all(test, not(feature = "mock")))]
mod profiler_tests {
    use super::*;

    struct CountingProfiler {
        calls: std::sync::atomic::AtomicU32,
    }

    impl ReportLayerTime for CountingProfiler {
        fn report_layer_time(&self, _layer_name: &str, _ms: f32) {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[test]
    fn profiler_cpp_bridge_allocates() {
        let inner = Box::new(CountingProfiler {
            calls: std::sync::atomic::AtomicU32::new(0),
        });
        let profiler = Profiler::new(inner).expect("profiler");
        assert!(!profiler.as_raw().is_null());
    }
}

#[cfg(test)]
mod stream_reader_v2_tests {
    use super::*;

    struct SliceReader {
        data: Vec<u8>,
        position: usize,
    }

    impl ReadStreamV2 for SliceReader {
        unsafe fn read(
            &mut self,
            destination: *mut std::ffi::c_void,
            byte_count: i64,
            _stream: *mut std::ffi::c_void,
        ) -> i64 {
            let Ok(byte_count) = usize::try_from(byte_count) else {
                return -1;
            };
            let byte_count = byte_count.min(self.data.len() - self.position);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data.as_ptr().add(self.position),
                    destination.cast::<u8>(),
                    byte_count,
                );
            }
            self.position += byte_count;
            byte_count as i64
        }

        fn seek(&mut self, offset: i64, position: SeekPosition) -> bool {
            let base = match position {
                SeekPosition::kSET => 0,
                SeekPosition::kCUR => self.position as i64,
                SeekPosition::kEND => self.data.len() as i64,
            };
            let Some(position) = base.checked_add(offset) else {
                return false;
            };
            let Ok(position) = usize::try_from(position) else {
                return false;
            };
            if position > self.data.len() {
                return false;
            }
            self.position = position;
            true
        }
    }

    #[test]
    fn callbacks_read_and_seek_the_rust_implementation() {
        let mut reader = StreamReaderV2::new(Box::new(SliceReader {
            data: b"engine-plan".to_vec(),
            position: 0,
        }))
        .expect("stream reader");
        let this = unsafe {
            reader.as_mut().get_unchecked_mut() as *mut StreamReaderV2 as *mut std::ffi::c_void
        };
        let mut output = [0_u8; 4];

        let read = unsafe {
            StreamReaderV2_read(
                this,
                output.as_mut_ptr().cast(),
                output.len() as i64,
                null_mut(),
            )
        };
        assert_eq!(read, 4);
        assert_eq!(&output, b"engi");

        assert!(unsafe { StreamReaderV2_seek(this, -4, SeekPosition::kEND as i32) });
        let read = unsafe {
            StreamReaderV2_read(
                this,
                output.as_mut_ptr().cast(),
                output.len() as i64,
                null_mut(),
            )
        };
        assert_eq!(read, 4);
        assert_eq!(&output, b"plan");
    }
}
