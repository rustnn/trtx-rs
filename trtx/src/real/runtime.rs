//! Real TensorRT runtime implementation

use std::ffi::CStr;
use std::marker::PhantomData;
use std::pin::Pin;
use std::ptr;

use cxx::UniquePtr;
use trtx_sys::nvinfer1::{
    self, DataType, ICudaEngine, IEngineInspector, IExecutionContext, LayerInformationFormat,
    TensorIOMode,
};

use crate::error::{Error, Result};
use crate::logger::Logger;

pub struct CudaEngine<'runtime> {
    inner: UniquePtr<ICudaEngine>,
    _runtime: PhantomData<&'runtime nvinfer1::IRuntime>,
}

impl<'engine> CudaEngine<'engine> {
    pub(crate) fn from_ptr(ptr: &'engine mut ICudaEngine) -> Self {
        Self {
            inner: unsafe { UniquePtr::from_raw(ptr) },
            _runtime: Default::default(),
        }
    }

    pub fn get_nb_io_tensors(&self) -> Result<i32> {
        Ok(self.inner.getNbIOTensors())
    }

    pub fn get_tensor_name(&self, index: i32) -> Result<String> {
        let name_ptr = self.inner.getIOTensorName(index);
        if name_ptr.is_null() {
            return Err(Error::InvalidArgument("Invalid tensor index".to_string()));
        }
        Ok(unsafe { CStr::from_ptr(name_ptr) }.to_str()?.to_string())
    }

    pub fn get_tensor_shape(&self, name: &str) -> Result<Vec<i64>> {
        let name_cstr = std::ffi::CString::new(name)?;
        let dims = unsafe { self.inner.getTensorShape(name_cstr.as_ptr()) };
        let nb_dims = dims.nbDims as usize;
        if nb_dims > 8 {
            return Err(Error::Runtime("Tensor has too many dimensions".to_string()));
        }
        Ok((0..nb_dims).map(|i| dims.d[i]).collect())
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorDataType`].
    pub fn get_tensor_data_type(&self, name: &str) -> Result<DataType> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorDataType(name_cstr.as_ptr()) })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getNbLayers`].
    pub fn get_nb_layers(&self) -> Result<i32> {
        Ok(self.inner.getNbLayers())
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getNbOptimizationProfiles`].
    pub fn get_nb_optimization_profiles(&self) -> Result<i32> {
        Ok(self.inner.getNbOptimizationProfiles())
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getNbAuxStreams`].
    pub fn get_nb_aux_streams(&self) -> Result<i32> {
        Ok(self.inner.getNbAuxStreams())
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorIOMode`].
    pub fn get_tensor_io_mode(&self, name: &str) -> Result<TensorIOMode> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorIOMode(name_cstr.as_ptr()) })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorLocation`].
    pub fn get_tensor_location(&self, name: &str) -> Result<trtx_sys::nvinfer1::TensorLocation> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorLocation(name_cstr.as_ptr()) })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorFormat`].
    pub fn get_tensor_format(&self, name: &str) -> Result<trtx_sys::nvinfer1::TensorFormat> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorFormat(name_cstr.as_ptr()) })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorFormat`] (profile variant).
    pub fn get_tensor_format_for_profile(
        &self,
        name: &str,
        profile_index: i32,
    ) -> Result<trtx_sys::nvinfer1::TensorFormat> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe {
            self.inner
                .getTensorFormat1(name_cstr.as_ptr(), profile_index)
        })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorFormatDesc`].
    pub fn get_tensor_format_desc(&self, name: &str) -> Result<String> {
        let name_cstr = std::ffi::CString::new(name)?;
        let ptr = unsafe { self.inner.getTensorFormatDesc(name_cstr.as_ptr()) };
        if ptr.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string())
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorFormatDesc`] (profile variant).
    pub fn get_tensor_format_desc_for_profile(
        &self,
        name: &str,
        profile_index: i32,
    ) -> Result<String> {
        let name_cstr = std::ffi::CString::new(name)?;
        let ptr = unsafe {
            self.inner
                .getTensorFormatDesc1(name_cstr.as_ptr(), profile_index)
        };
        if ptr.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string())
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorVectorizedDim`].
    pub fn get_tensor_vectorized_dim(&self, name: &str) -> Result<i32> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorVectorizedDim(name_cstr.as_ptr()) })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorVectorizedDim`] (profile variant).
    pub fn get_tensor_vectorized_dim_for_profile(
        &self,
        name: &str,
        profile_index: i32,
    ) -> Result<i32> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe {
            self.inner
                .getTensorVectorizedDim1(name_cstr.as_ptr(), profile_index)
        })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorBytesPerComponent`].
    pub fn get_tensor_bytes_per_component(&self, name: &str) -> Result<i32> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorBytesPerComponent(name_cstr.as_ptr()) })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorBytesPerComponent`] (profile variant).
    pub fn get_tensor_bytes_per_component_for_profile(
        &self,
        name: &str,
        profile_index: i32,
    ) -> Result<i32> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe {
            self.inner
                .getTensorBytesPerComponent1(name_cstr.as_ptr(), profile_index)
        })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorComponentsPerElement`].
    pub fn get_tensor_components_per_element(&self, name: &str) -> Result<i32> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorComponentsPerElement(name_cstr.as_ptr()) })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorComponentsPerElement`] (profile variant).
    pub fn get_tensor_components_per_element_for_profile(
        &self,
        name: &str,
        profile_index: i32,
    ) -> Result<i32> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe {
            self.inner
                .getTensorComponentsPerElement1(name_cstr.as_ptr(), profile_index)
        })
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::createEngineInspector`].
    /// Returns an inspector that can print layer and engine information (e.g. JSON or one-line format).
    pub fn create_engine_inspector(&self) -> Result<EngineInspector<'_>> {
        let inspector = self.inner.createEngineInspector();
        let inspector = unsafe {
            inspector
                .as_mut()
                .ok_or_else(|| Error::Runtime("Failed to create engine inspector".to_string()))?
        };
        Ok(EngineInspector {
            inner: unsafe { Pin::new_unchecked(inspector) },
        })
    }

    /// Returns the data type of the tensor (e.g. kFLOAT, kHALF).
    /// Required for correct buffer sizing and f32/f16 conversion when I/O uses half precision.
    pub fn get_tensor_dtype(&self, name: &str) -> Result<trtx_sys::DataType> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorDataType(name_cstr.as_ptr()).into() })
    }

    pub fn create_execution_context(&'_ mut self) -> Result<ExecutionContext<'engine>> {
        let context_ptr = self.inner.pin_mut().createExecutionContext(
            trtx_sys::nvinfer1::ExecutionContextAllocationStrategy::kSTATIC,
        );
        if context_ptr.is_null() {
            return Err(Error::Runtime(
                "Failed to create execution context".to_string(),
            ));
        }
        Ok(ExecutionContext {
            inner: context_ptr as *mut _,
            _engine: std::marker::PhantomData,
        })
    }
}

/// Engine inspector for layer/engine information (real mode).
/// See [`trtx_sys::nvinfer1::IEngineInspector`].
pub struct EngineInspector<'engine> {
    inner: Pin<&'engine mut IEngineInspector>,
}

impl EngineInspector<'_> {
    /// Returns layer information for the given layer index in the requested format.
    /// See [`trtx_sys::nvinfer1::IEngineInspector::getLayerInformation`].
    pub fn get_layer_information(
        &mut self,
        layer_index: i32,
        format: LayerInformationFormat,
    ) -> Result<String> {
        let ptr = self.inner.getLayerInformation(layer_index, format);
        Ok(if ptr.is_null() {
            return Err(Error::Runtime(
                "Could not get layer information".to_string(),
            ));
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string()
        })
    }

    /// Returns engine information in the requested format.
    /// See [`trtx_sys::nvinfer1::IEngineInspector::getEngineInformation`].
    pub fn get_engine_information(&self, format: LayerInformationFormat) -> Result<String> {
        let ptr = self.inner.getEngineInformation(format);
        Ok(if ptr.is_null() {
            return Err(Error::Runtime(
                "Could not get layer information".to_string(),
            ));
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string()
        })
    }
}

impl Drop for EngineInspector<'_> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(Pin::get_unchecked_mut(self.inner.as_mut()) as *mut IEngineInspector);
        };
    }
}

/// Execution context (real mode)
pub struct ExecutionContext<'a> {
    inner: *mut std::ffi::c_void,
    _engine: std::marker::PhantomData<&'a CudaEngine<'a>>,
}

impl<'a> ExecutionContext<'a> {
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
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid execution context".to_string()));
        }
        let name_cstr = std::ffi::CString::new(name)?;
        let success = crate::autocxx_helpers::cast_and_pin::<IExecutionContext>(self.inner)
            .setTensorAddress(name_cstr.as_ptr(), data as *mut _);
        if !success {
            return Err(Error::Runtime("Failed to set tensor address".to_string()));
        }
        Ok(())
    }

    /// Enqueues inference on the given CUDA stream.
    ///
    /// # Safety
    /// `cuda_stream` must be a valid CUDA stream, and all tensor addresses must
    /// point to valid device memory.
    pub unsafe fn enqueue_v3(&mut self, cuda_stream: *mut std::ffi::c_void) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::Runtime("Invalid execution context".to_string()));
        }
        let success = crate::autocxx_helpers::cast_and_pin::<IExecutionContext>(self.inner)
            .enqueueV3(cuda_stream as *mut _);
        if !success {
            return Err(Error::Runtime("Failed to enqueue inference".to_string()));
        }
        Ok(())
    }
}

impl Drop for ExecutionContext<'_> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                trtx_sys::delete_context(self.inner);
            }
        }
    }
}

unsafe impl Send for ExecutionContext<'_> {}

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
