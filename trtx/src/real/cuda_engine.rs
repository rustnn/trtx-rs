use std::{ffi::CStr, marker::PhantomData};

use crate::error::PropertySetAttempt;
use crate::{
    real::{engine_inspector::EngineInspector, host_memory::HostMemory},
    Error, ExecutionContext, Result,
};
use autocxx::cxx::UniquePtr;
use trtx_sys::{
    nvinfer1::{self, DataType, ICudaEngine, TensorIOMode},
    SerializationFlag,
};

pub struct SerializationConfig<'cuda_engine> {
    inner: UniquePtr<nvinfer1::ISerializationConfig>,
    _runtime: PhantomData<&'cuda_engine nvinfer1::ICudaEngine>,
}
impl SerializationConfig<'_> {
    pub fn get_flag(&self, flag: SerializationFlag) -> bool {
        self.inner.getFlag(flag.into())
    }
    pub fn get_flags(&self) -> u32 {
        self.inner.getFlags()
    }
    pub fn set_flag(&mut self, flag: SerializationFlag) -> Result<()> {
        if self.inner.pin_mut().setFlag(flag.into()) {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(
                PropertySetAttempt::SerializationFlag,
            ))
        }
    }
    pub fn set_flags(&mut self, flags: u32) -> Result<()> {
        if self.inner.pin_mut().setFlags(flags) {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(
                PropertySetAttempt::SerializationFlag,
            ))
        }
    }
    pub fn clear_flag(&mut self, flag: SerializationFlag) -> Result<()> {
        if self.inner.pin_mut().clearFlag(flag.into()) {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(
                PropertySetAttempt::SerializationFlag,
            ))
        }
    }
}

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
        #[cfg(not(feature = "mock"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            Ok(unsafe { self.inner.getTensorBytesPerComponent(name_cstr.as_ptr()) })
        }
        #[cfg(feature = "mock")]
        Ok(42)
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorBytesPerComponent`] (profile variant).
    pub fn get_tensor_bytes_per_component_for_profile(
        &self,
        name: &str,
        profile_index: i32,
    ) -> Result<i32> {
        if !self.inner.is_null() {
            let name_cstr = std::ffi::CString::new(name)?;
            Ok(unsafe {
                self.inner
                    .getTensorBytesPerComponent1(name_cstr.as_ptr(), profile_index)
            })
        } else {
            Ok(0)
        }
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorComponentsPerElement`].
    pub fn get_tensor_components_per_element(&self, name: &str) -> Result<i32> {
        #[cfg(not(feature = "mock"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            Ok(unsafe { self.inner.getTensorComponentsPerElement(name_cstr.as_ptr()) })
        }
        #[cfg(feature = "mock")]
        {
            Ok(42)
        }
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::getTensorComponentsPerElement`] (profile variant).
    pub fn get_tensor_components_per_element_for_profile(
        &self,
        name: &str,
        profile_index: i32,
    ) -> Result<i32> {
        #[cfg(not(feature = "mock"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            Ok(unsafe {
                self.inner
                    .getTensorComponentsPerElement1(name_cstr.as_ptr(), profile_index)
            })
        }
        #[cfg(feature = "mock")]
        {
            Ok(42)
        }
    }

    /// See [`trtx_sys::nvinfer1::ICudaEngine::createEngineInspector`].
    /// Returns an inspector that can print layer and engine information (e.g. JSON or one-line format).
    pub fn create_engine_inspector(&self) -> Result<EngineInspector<'_>> {
        #[cfg(not(feature = "mock"))]
        {
            use crate::real::engine_inspector::EngineInspector;

            let inspector = self.inner.createEngineInspector();
            let inspector = unsafe {
                inspector.as_mut().ok_or_else(|| {
                    Error::Runtime("Failed to create engine inspector".to_string())
                })?
            };
            Ok(EngineInspector {
                inner: unsafe { UniquePtr::from_raw(inspector) },
                _engine: Default::default(),
            })
        }
        #[cfg(feature = "mock")]
        {
            Ok(EngineInspector {
                inner: UniquePtr::null(),
                _engine: Default::default(),
            })
        }
    }

    /// Returns the data type of the tensor (e.g. kFLOAT, kHALF).
    /// Required for correct buffer sizing and f32/f16 conversion when I/O uses half precision.
    pub fn get_tensor_dtype(&self, name: &str) -> Result<trtx_sys::DataType> {
        #[cfg(not(feature = "mock"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            Ok(unsafe { self.inner.getTensorDataType(name_cstr.as_ptr()).into() })
        }
        #[cfg(feature = "mock")]
        Ok(trtx_sys::DataType::kFLOAT)
    }

    pub fn create_execution_context(&'_ mut self) -> Result<ExecutionContext<'engine>> {
        #[cfg(not(feature = "mock"))]
        {
            use crate::ExecutionContext;

            let context_ptr = self.inner.pin_mut().createExecutionContext(
                trtx_sys::nvinfer1::ExecutionContextAllocationStrategy::kSTATIC,
            );
            Ok(unsafe { ExecutionContext::from_ptr(context_ptr)? })
        }
        #[cfg(feature = "mock")]
        Ok(unsafe { ExecutionContext::from_ptr(std::ptr::null_mut())? })
    }

    pub fn create_serialization_config(&mut self) -> Result<SerializationConfig<'engine>> {
        let config = unsafe {
            self.inner
                .pin_mut()
                .createSerializationConfig()
                .as_mut()
                .ok_or_else(|| Error::Runtime("SerializationConfig creation failed".to_string()))?
        };
        Ok(SerializationConfig {
            inner: unsafe { UniquePtr::from_raw(config) },
            _runtime: Default::default(),
        })
    }

    // TODO: highlevel type for ISerializationConfig
    /// See [nvinfer1::ICudaEngine::serializeWithConfig]
    pub fn serialize_with_config(
        &'_ self,
        config: &mut SerializationConfig,
    ) -> Result<HostMemory<'engine>> {
        if !cfg!(feature = "mock") {
            let host_mem = unsafe {
                self.inner
                    .serializeWithConfig(config.inner.pin_mut())
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Runtime("Failed to serialize ICudaEngine with config".to_string())
                    })?
            };
            Ok(unsafe { HostMemory::from_raw(host_mem) })
        } else {
            Ok(unsafe { HostMemory::from_raw(std::ptr::null_mut()) })
        }
    }
}
