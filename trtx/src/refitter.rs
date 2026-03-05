use std::ffi::{CStr, CString};
use std::marker::PhantomData;

use crate::{
    error::{Error, Result},
    CudaEngine, Logger,
};
use autocxx::cxx::UniquePtr;
use trtx_sys::{nvinfer1, ErrorRecorder};

pub struct Refitter<'logger, 'engine> {
    inner: UniquePtr<nvinfer1::IRefitter>,
    error_recorder: Option<ErrorRecorder>,
    _logger: PhantomData<&'logger Logger>,
    _engine: PhantomData<&'engine CudaEngine<'engine>>,
}

impl<'logger, 'engine> Refitter<'logger, 'engine> {
    #[cfg(not(feature = "link_tensorrt_rtx"))]
    #[cfg(not(feature = "dlopen_tensorrt_rtx"))]
    pub fn new(_cuda_engine: &'engine CudaEngine, _logger: &'logger Logger) -> Result<Self> {
        Err(Error::TrtRtxLibraryNotLoaded)
    }

    #[cfg(any(feature = "link_tensorrt_rtx", feature = "dlopen_tensorrt_rtx"))]
    pub fn new(cuda_engine: &'engine CudaEngine, logger: &'logger Logger) -> Result<Self> {
        #[cfg(not(feature = "mock"))]
        {
            let logger_ptr = logger.as_logger_ptr();
            let engine_ptr = cuda_engine.inner.as_mut_ptr() as *mut std::ffi::c_void;
            let refitter = {
                #[cfg(feature = "link_tensorrt_rtx")]
                unsafe {
                    trtx_sys::create_infer_refitter(engine_ptr, logger_ptr)
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
                    let create_infer_refitter: Symbol<
                        fn(*mut c_void, *mut c_void, u32) -> *mut nvinfer1::IRefitter,
                    > = lock
                        .as_ref()
                        .ok_or(Error::TrtRtxLibraryNotLoaded)?
                        .get(b"createInferRefitter_INTERNAL")?;
                    create_infer_refitter(engine_ptr, logger_ptr, trtx_sys::get_tensorrt_version())
                }
            };
            if refitter.is_null() {
                return Err(Error::Runtime("Failed to create refitter".to_string()));
            }
            Ok(Self {
                inner: unsafe { UniquePtr::from_raw(refitter) },
                error_recorder: None,
                _engine: Default::default(),
                _logger: Default::default(),
            })
        }
        #[cfg(feature = "mock")]
        Ok(Refitter {
            inner: UniquePtr::null(),
            _engine: Default::default(),
            _logger: Default::default(),
        })
    }

    /// Specify new weights for a layer of given name and role.
    pub fn set_weights(
        &mut self,
        layer_name: &str,
        role: nvinfer1::WeightsRole,
        weights: nvinfer1::Weights,
    ) -> Result<()> {
        let name_cstr = CString::new(layer_name)?;
        if unsafe {
            self.inner
                .pin_mut()
                .setWeights(name_cstr.as_ptr(), role, weights)
        } {
            Ok(())
        } else {
            Err(Error::Runtime(
                "setWeights rejected (invalid layer/role/count/type)".to_string(),
            ))
        }
    }

    /// Refit the associated engine. Returns an error if validation fails or there are missing weights.
    pub fn refit_cuda_engine(&mut self) -> Result<()> {
        if self.inner.pin_mut().refitCudaEngine() {
            Ok(())
        } else {
            Err(Error::Runtime(
                "refitCudaEngine failed (validation or getMissingWeights != 0)".to_string(),
            ))
        }
    }

    /// Get descriptions of missing weights (layer name + role). Call with a size to limit results.
    pub fn get_missing(&self, max_count: i32) -> Result<Vec<(String, nvinfer1::WeightsRole)>> {
        let n = max_count.max(0) as usize;
        let mut layer_names: Vec<*const std::ffi::c_char> = vec![std::ptr::null(); n];
        let mut roles: Vec<i32> = vec![0; n];
        let refitter_ptr = self.refitter_ptr();
        let count = unsafe {
            trtx_sys::trtx_refitter_get_missing(
                refitter_ptr,
                n as i32,
                layer_names.as_mut_ptr(),
                roles.as_mut_ptr(),
            )
        };
        let count = count.max(0) as usize;
        let mut out = Vec::with_capacity(count);
        for i in 0..count.min(layer_names.len()) {
            let ptr = layer_names[i];
            if ptr.is_null() {
                break;
            }
            let s = unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string();
            let role = unsafe { std::mem::transmute::<i32, nvinfer1::WeightsRole>(roles[i]) };
            out.push((s, role));
        }
        Ok(out)
    }

    /// Get descriptions of all weights that could be refit (layer name + role).
    pub fn get_all(&self, max_count: i32) -> Result<Vec<(String, nvinfer1::WeightsRole)>> {
        let n = max_count.max(0) as usize;
        let mut layer_names: Vec<*const std::ffi::c_char> = vec![std::ptr::null(); n];
        let mut roles: Vec<i32> = vec![0; n];
        let refitter_ptr = self.refitter_ptr();
        let count = unsafe {
            trtx_sys::trtx_refitter_get_all(
                refitter_ptr,
                n as i32,
                layer_names.as_mut_ptr(),
                roles.as_mut_ptr(),
            )
        };
        let count = count.max(0) as usize;
        let mut out = Vec::with_capacity(count);
        for i in 0..count.min(layer_names.len()) {
            let ptr = layer_names[i];
            if ptr.is_null() {
                break;
            }
            let s = unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string();
            let role = unsafe { std::mem::transmute::<i32, nvinfer1::WeightsRole>(roles[i]) };
            out.push((s, role));
        }
        Ok(out)
    }

    /// See [nvinfer1::IRefitter::setErrorRecorder]
    pub fn set_error_recorder(&mut self, error_recorder: ErrorRecorder) {
        self.error_recorder = Some(error_recorder);
        unsafe {
            self.inner.pin_mut().setErrorRecorder(
                self.error_recorder
                    .as_mut()
                    .unwrap()
                    .pin_mut()
                    .get_unchecked_mut(),
            )
        };
    }

    /// Get the assigned error recorder, or null if none.
    pub fn get_error_recorder(&self) -> *mut nvinfer1::IErrorRecorder {
        self.inner.getErrorRecorder()
    }

    /// Specify new weights by name (host location by default).
    pub fn set_named_weights(&mut self, name: &str, weights: nvinfer1::Weights) -> Result<()> {
        let name_cstr = CString::new(name)?;
        if unsafe {
            self.inner
                .pin_mut()
                .setNamedWeights(name_cstr.as_ptr(), weights)
        } {
            Ok(())
        } else {
            Err(Error::Runtime(
                "setNamedWeights rejected (invalid name/count/type)".to_string(),
            ))
        }
    }

    /// Get names of missing weights.
    pub fn get_missing_weights(&self, max_count: i32) -> Result<Vec<String>> {
        let n = max_count.max(0) as usize;
        let mut names: Vec<*const std::ffi::c_char> = vec![std::ptr::null(); n];
        let count = unsafe {
            trtx_sys::trtx_refitter_get_missing_weights(
                self.refitter_ptr(),
                n as i32,
                names.as_mut_ptr(),
            )
        };
        let count = count.max(0) as usize;
        let mut out = Vec::with_capacity(count);
        for i in 0..count.min(names.len()) {
            let ptr = names[i];
            if ptr.is_null() {
                break;
            }
            out.push(unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string());
        }
        Ok(out)
    }

    /// Get names of all weights that could be refit.
    pub fn get_all_weights(&self, max_count: i32) -> Result<Vec<String>> {
        let n = max_count.max(0) as usize;
        let mut names: Vec<*const std::ffi::c_char> = vec![std::ptr::null(); n];
        let count = unsafe {
            trtx_sys::trtx_refitter_get_all_weights(
                self.refitter_ptr(),
                n as i32,
                names.as_mut_ptr(),
            )
        };
        let count = count.max(0) as usize;
        let mut out = Vec::with_capacity(count);
        for i in 0..count.min(names.len()) {
            let ptr = names[i];
            if ptr.is_null() {
                break;
            }
            out.push(unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string());
        }
        Ok(out)
    }

    /// Raw pointer to the underlying IRefitter (for C wrappers). Caller must not use after Refitter is dropped.
    fn refitter_ptr(&self) -> *mut std::ffi::c_void {
        self.inner.as_ptr() as *mut std::ffi::c_void
    }

    /// Get the logger with which the refitter was created. Returns raw pointer.
    pub fn get_logger(&self) -> *mut nvinfer1::ILogger {
        self.inner.getLogger()
    }

    /// Set the maximum number of threads used by the refitter.
    pub fn set_max_threads(&mut self, max_threads: i32) -> Result<()> {
        if self.inner.pin_mut().setMaxThreads(max_threads) {
            Ok(())
        } else {
            Err(Error::InvalidArgument("setMaxThreads failed".to_string()))
        }
    }

    /// Get the maximum number of threads that can be used by the refitter.
    pub fn get_max_threads(&self) -> i32 {
        self.inner.getMaxThreads()
    }

    /// Specify new weights by name with explicit host/device location.
    pub fn set_named_weights_with_location(
        &mut self,
        name: &str,
        weights: nvinfer1::Weights,
        location: nvinfer1::TensorLocation,
    ) -> Result<()> {
        let name_cstr = CString::new(name)?;
        if unsafe {
            self.inner
                .pin_mut()
                .setNamedWeights1(name_cstr.as_ptr(), weights, location)
        } {
            Ok(())
        } else {
            Err(Error::Runtime(
                "setNamedWeights (with location) rejected".to_string(),
            ))
        }
    }

    /// Get weights currently associated with the given name.
    pub fn get_named_weights(&self, weights_name: &str) -> nvinfer1::Weights {
        let name_cstr = CString::new(weights_name).expect("name contains null");
        unsafe { self.inner.getNamedWeights(name_cstr.as_ptr()) }
    }

    /// Get the location for weights associated with the given name.
    pub fn get_weights_location(&self, weights_name: &str) -> nvinfer1::TensorLocation {
        let name_cstr = CString::new(weights_name).expect("name contains null");
        unsafe { self.inner.getWeightsLocation(name_cstr.as_ptr()) }
    }

    /// Unset weights for the given name. Returns false if they were never set.
    pub fn unset_named_weights(&mut self, weights_name: &str) -> bool {
        let name_cstr = CString::new(weights_name).expect("name contains null");
        unsafe { self.inner.pin_mut().unsetNamedWeights(name_cstr.as_ptr()) }
    }

    /// Set whether to validate weights during refitting (default true).
    pub fn set_weights_validation(&mut self, weights_validation: bool) {
        self.inner
            .pin_mut()
            .setWeightsValidation(weights_validation);
    }

    /// Get whether weights validation is enabled during refitting.
    pub fn get_weights_validation(&self) -> bool {
        self.inner.getWeightsValidation()
    }

    /// Enqueue weights refitting on the given CUDA stream.
    pub unsafe fn refit_cuda_engine_async(
        &mut self,
        cuda_stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        if self
            .inner
            .pin_mut()
            .refitCudaEngineAsync(cuda_stream as *mut _)
        {
            Ok(())
        } else {
            Err(Error::Runtime(
                "refitCudaEngineAsync failed (validation or getMissingWeights != 0)".to_string(),
            ))
        }
    }

    /// Get the weights prototype (type and count) for the given name. Values pointer is null.
    pub fn get_weights_prototype(&self, weights_name: &str) -> nvinfer1::Weights {
        let name_cstr = CString::new(weights_name).expect("name contains null");
        unsafe { self.inner.getWeightsPrototype(name_cstr.as_ptr()) }
    }
}
