use crate::ffi;
use ::autocxx::subclass::*;
use std::ffi::CStr;

use crate::nvinfer1;

/// Rust version of the [nvinfer1::IProgressMonitor]
///
/// Put into a [ProgressMonitor] to subclass [nvinfer1::IProgressMonitor]
pub trait HandleProgress {
    /// See [nvinfer::IProgressMonitor::phaseStart]
    fn phase_start(&mut self, phase_name: &str, parent_phase: &str, num_steps: i32);
    /// See [nvinfer::IProgressMonitor::stepComplete]. Return whether to continue building or cancel
    fn step_complete(&mut self, phase_name: &str, step: i32) -> std::ops::ControlFlow<()>;
    /// See [nvinfer::IProgressMonitor::phaseFinish]
    fn phase_finish(&mut self, phase_name: &str);
}

#[subclass]
#[derive(Default)]
/// Subclasses [nvinfer1::IProgressMonitor]
///
/// Construct a object with a dyn [HandleProgress] to implement
/// [nvinfer1::IProgressMonitor] from Rust
pub struct ProgressMonitor {
    inner: Option<Box<dyn HandleProgress>>,
}

impl ProgressMonitor {
    pub fn new(inner: Box<dyn HandleProgress>) -> Self {
        Self {
            inner: Some(inner),
            ..Default::default()
        }
    }
}

impl nvinfer1::IProgressMonitor_methods for ProgressMonitor {
    unsafe fn phaseStart(
        &mut self,
        phaseName: *const ::std::os::raw::c_char,
        parentPhase: *const ::std::os::raw::c_char,
        nbSteps: i32,
    ) {
        let phase_name = CStr::from_ptr(phaseName);
        let parent_phase = CStr::from_ptr(parentPhase);
        self.inner
            .as_mut()
            .expect("construction only possible with Some")
            .phase_start(
                &phase_name.to_string_lossy(),
                &parent_phase.to_string_lossy(),
                nbSteps,
            );
    }
    unsafe fn stepComplete(&mut self, phaseName: *const ::std::os::raw::c_char, step: i32) -> bool {
        let phase_name = CStr::from_ptr(phaseName);
        self.inner
            .as_mut()
            .expect("construction only possible with Some")
            .step_complete(&phase_name.to_string_lossy(), step)
            .is_continue()
    }
    unsafe fn phaseFinish(&mut self, phaseName: *const ::std::os::raw::c_char) {
        let phase_name = CStr::from_ptr(phaseName);
        self.inner
            .as_mut()
            .expect("construction only possible with Some")
            .phase_finish(&phase_name.to_string_lossy());
    }
}

#[subclass]
#[derive(Default)]
pub struct GpuAllocator {
    inner: Option<Box<dyn AllocateGpu>>,
}

impl GpuAllocator {
    pub fn new(inner: Box<dyn AllocateGpu>) -> Self {
        Self {
            inner: Some(inner),
            ..Default::default()
        }
    }
}

impl nvinfer1::IGpuAllocator_methods for GpuAllocator {
    fn allocate(&mut self, size: u64, alignment: u64, flags: u32) -> *mut autocxx::c_void {
        self.inner
            .as_mut()
            .unwrap()
            .allocate(size, alignment, flags)
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
        unsafe { self.inner.as_mut().unwrap().deallocate(data) }
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

pub trait AllocateGpu {
    fn allocate(&mut self, size: u64, alignment: u64, flags: u32) -> *mut autocxx::c_void;
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
    unsafe fn deallocate(&mut self, data: *mut autocxx::c_void) -> bool;
    unsafe fn deallocate_async(
        &mut self,
        data: *mut autocxx::c_void,
        cuda_stream: *mut crate::ffi::CUstream_st,
    ) -> bool;
}
