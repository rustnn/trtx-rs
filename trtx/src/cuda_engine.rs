//! CUDA engine and serialization config.
//!
//! [`CudaEngine`] wraps [`nvinfer1::ICudaEngine`] (C++ [`nvinfer1::ICudaEngine`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_cuda_engine.html)).
//! [`SerializationConfig`] wraps [`nvinfer1::ISerializationConfig`] (C++ [`nvinfer1::ISerializationConfig`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_serialization_config.html)).

use std::{ffi::CStr, marker::PhantomData};

use crate::engine_inspector::EngineInspector;
use crate::error::PropertySetAttempt;
use crate::host_memory::HostMemory;
use crate::{DataType, Error, ExecutionContext, Result};
use autocxx::cxx::UniquePtr;
use trtx_sys::{
    nvinfer1::{self, ICudaEngine},
    SerializationFlag, TensorIOMode,
};
use trtx_sys::{TensorFormat, TensorLocation};

/// [`nvinfer1::ISerializationConfig`] — C++ [`nvinfer1::ISerializationConfig`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_serialization_config.html).
pub struct SerializationConfig<'cuda_engine> {
    inner: UniquePtr<nvinfer1::ISerializationConfig>,
    _runtime: PhantomData<&'cuda_engine nvinfer1::ICudaEngine>,
}
impl SerializationConfig<'_> {
    /// See [nvinfer1::ISerializationConfig::getFlag]
    pub fn flag(&self, flag: SerializationFlag) -> bool {
        self.inner.getFlag(flag.into())
    }
    /// See [nvinfer1::SerializationConfig::getFlags]
    pub fn flags(&self) -> u32 {
        self.inner.getFlags()
    }
    /// See [nvinfer1::ISerializationConfig::setFlag]
    pub fn set_flag(&mut self, flag: SerializationFlag) -> Result<()> {
        if self.inner.pin_mut().setFlag(flag.into()) {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(
                PropertySetAttempt::SerializationFlag,
            ))
        }
    }
    /// See [nvinfer1::ISerializationConfig::setFlags]
    pub fn set_flags(&mut self, flags: u32) -> Result<()> {
        if self.inner.pin_mut().setFlags(flags) {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(
                PropertySetAttempt::SerializationFlag,
            ))
        }
    }
    /// See [nvinfer1::ISerializationConfig::clearFlag]
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

/// [nvinfer1::ICudaEngine] — C++ [`nvinfer1::ICudaEngine`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_cuda_engine.html).
pub struct CudaEngine<'runtime> {
    pub(crate) inner: UniquePtr<ICudaEngine>,
    _runtime: PhantomData<&'runtime nvinfer1::IRuntime>,
}

impl<'engine> CudaEngine<'engine> {
    pub(crate) unsafe fn from_ptr(ptr: *mut ICudaEngine) -> Self {
        Self {
            inner: unsafe { UniquePtr::from_raw(ptr) },
            _runtime: Default::default(),
        }
    }

    #[deprecated = "use nb_io_tensors instead"]
    pub fn get_nb_io_tensors(&self) -> Result<i32> {
        self.nb_io_tensors()
    }
    #[deprecated = "use tensor_shape instead"]
    pub fn get_tensor_shape(&self, name: &str) -> Result<Vec<i64>> {
        self.tensor_shape(name)
    }
    #[deprecated = "use io_tensor_name instead"]
    pub fn get_tensor_name(&self, index: i32) -> Result<String> {
        self.io_tensor_name(index)
    }
    #[deprecated = "use tensor_data_type instead"]
    pub fn get_tensor_dtype(&self, name: &str) -> Result<DataType> {
        self.tensor_data_type(name)
    }

    /// See [ICudaEngine::getName]
    pub fn name(&self) -> Result<String> {
        let ptr = self.inner.getName();
        if ptr.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string())
    }

    /// See [`nvinfer1::ICudaEngine::getNbIOTensors`].
    pub fn nb_io_tensors(&self) -> Result<i32> {
        if cfg!(feature = "mock_runtime") {
            Ok(0)
        } else {
            Ok(self.inner.getNbIOTensors())
        }
    }

    /// See [`nvinfer1::ICudaEngine::getIOTensorName`].
    pub fn io_tensor_name(&self, index: i32) -> Result<String> {
        if cfg!(feature = "mock_runtime") {
            Ok("mock_runtime".to_string())
        } else {
            let name_ptr = self.inner.getIOTensorName(index);
            if name_ptr.is_null() {
                return Err(Error::InvalidArgument("Invalid tensor index".to_string()));
            }
            Ok(unsafe { CStr::from_ptr(name_ptr) }.to_str()?.to_string())
        }
    }

    /// See [`nvinfer1::ICudaEngine::getTensorShape`].
    pub fn tensor_shape(&self, name: &str) -> Result<Vec<i64>> {
        let name_cstr = std::ffi::CString::new(name)?;
        let dims = unsafe { self.inner.getTensorShape(name_cstr.as_ptr()) };
        let nb_dims = dims.nbDims as usize;
        if nb_dims > 8 {
            return Err(Error::Runtime("Tensor has too many dimensions".to_string()));
        }
        Ok((0..nb_dims).map(|i| dims.d[i]).collect())
    }

    /// See [`nvinfer1::ICudaEngine::getTensorDataType`].
    pub fn tensor_data_type(&self, name: &str) -> Result<DataType> {
        if cfg!(not(feature = "mock_runtime")) {
            let name_cstr = std::ffi::CString::new(name)?;
            Ok(unsafe { self.inner.getTensorDataType(name_cstr.as_ptr()) }.into())
        } else {
            Ok(DataType::kFLOAT)
        }
    }

    /// See [`nvinfer1::ICudaEngine::getNbLayers`].
    pub fn nb_layers(&self) -> Result<i32> {
        Ok(self.inner.getNbLayers())
    }

    /// See [`nvinfer1::ICudaEngine::getNbOptimizationProfiles`].
    pub fn nb_optimization_profiles(&self) -> Result<i32> {
        Ok(self.inner.getNbOptimizationProfiles())
    }

    /// See [`nvinfer1::ICudaEngine::getNbAuxStreams`].
    pub fn nb_aux_streams(&self) -> Result<i32> {
        Ok(self.inner.getNbAuxStreams())
    }

    /// See [`nvinfer1::ICudaEngine::getTensorIOMode`].
    pub fn tensor_io_mode(&self, name: &str) -> Result<TensorIOMode> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorIOMode(name_cstr.as_ptr()).into() })
    }

    /// See [`nvinfer1::ICudaEngine::getTensorLocation`].
    pub fn tensor_location(&self, name: &str) -> Result<TensorLocation> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorLocation(name_cstr.as_ptr()).into() })
    }

    /// See [`nvinfer1::ICudaEngine::getTensorFormat`].
    pub fn tensor_format(&self, name: &str) -> Result<TensorFormat> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorFormat(name_cstr.as_ptr()).into() })
    }

    /// See [`nvinfer1::ICudaEngine::getTensorFormat`] (profile variant).
    pub fn tensor_format_for_profile(
        &self,
        name: &str,
        profile_index: i32,
    ) -> Result<nvinfer1::TensorFormat> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe {
            self.inner
                .getTensorFormat1(name_cstr.as_ptr(), profile_index)
        })
    }

    /// See [`nvinfer1::ICudaEngine::getTensorFormatDesc`].
    pub fn tensor_format_desc(&self, name: &str) -> Result<String> {
        let name_cstr = std::ffi::CString::new(name)?;
        let ptr = unsafe { self.inner.getTensorFormatDesc(name_cstr.as_ptr()) };
        if ptr.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string())
    }

    /// See [`nvinfer1::ICudaEngine::getTensorFormatDesc`] (profile variant).
    pub fn tensor_format_desc_for_profile(&self, name: &str, profile_index: i32) -> Result<String> {
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

    /// See [`nvinfer1::ICudaEngine::getTensorVectorizedDim`].
    pub fn tensor_vectorized_dim(&self, name: &str) -> Result<i32> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.getTensorVectorizedDim(name_cstr.as_ptr()) })
    }

    /// See [`nvinfer1::ICudaEngine::getTensorVectorizedDim`] (profile variant).
    pub fn tensor_vectorized_dim_for_profile(&self, name: &str, profile_index: i32) -> Result<i32> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe {
            self.inner
                .getTensorVectorizedDim1(name_cstr.as_ptr(), profile_index)
        })
    }

    /// See [`nvinfer1::ICudaEngine::getTensorBytesPerComponent`].
    pub fn tensor_bytes_per_component(&self, name: &str) -> Result<i32> {
        #[cfg(not(feature = "mock_runtime"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            Ok(unsafe { self.inner.getTensorBytesPerComponent(name_cstr.as_ptr()) })
        }
        #[cfg(feature = "mock_runtime")]
        Ok(42)
    }

    /// See [`nvinfer1::ICudaEngine::getTensorBytesPerComponent`] (profile variant).
    pub fn tensor_bytes_per_component_for_profile(
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

    /// See [`nvinfer1::ICudaEngine::getTensorComponentsPerElement`].
    pub fn tensor_components_per_element(&self, name: &str) -> Result<i32> {
        #[cfg(not(feature = "mock_runtime"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            Ok(unsafe { self.inner.getTensorComponentsPerElement(name_cstr.as_ptr()) })
        }
        #[cfg(feature = "mock_runtime")]
        {
            Ok(42)
        }
    }

    /// See [`nvinfer1::ICudaEngine::getTensorComponentsPerElement`] (profile variant).
    pub fn tensor_components_per_element_for_profile(
        &self,
        name: &str,
        profile_index: i32,
    ) -> Result<i32> {
        #[cfg(not(feature = "mock_runtime"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            Ok(unsafe {
                self.inner
                    .getTensorComponentsPerElement1(name_cstr.as_ptr(), profile_index)
            })
        }
        #[cfg(feature = "mock_runtime")]
        {
            Ok(42)
        }
    }

    /// See [`nvinfer1::ICudaEngine::createEngineInspector`].
    /// Returns an inspector that can print layer and engine information (e.g. JSON or one-line format).
    pub fn create_engine_inspector(&self) -> Result<EngineInspector<'_>> {
        #[cfg(not(feature = "mock_runtime"))]
        {
            use crate::engine_inspector::EngineInspector;

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
        #[cfg(feature = "mock_runtime")]
        {
            Ok(EngineInspector {
                inner: UniquePtr::null(),
                _engine: Default::default(),
            })
        }
    }

    #[deprecated = "use tensor_data_type instead"]
    pub fn tensor_dtype(&self, name: &str) -> Result<DataType> {
        self.tensor_data_type(name)
    }

    /// See [nvinfer1::ICudaEngine::createExecutionContext]
    pub fn create_execution_context(&'_ mut self) -> Result<ExecutionContext<'engine>> {
        #[cfg(not(feature = "mock_runtime"))]
        {
            use crate::ExecutionContext;

            let context_ptr = self
                .inner
                .pin_mut()
                .createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kSTATIC);
            Ok(unsafe { ExecutionContext::from_ptr(context_ptr)? })
        }
        #[cfg(feature = "mock_runtime")]
        Ok(unsafe { ExecutionContext::from_ptr(std::ptr::null_mut())? })
    }

    /// See [nvinfer1::ICudaEngine::createSerializationConfig]
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

    /// See [nvinfer1::ICudaEngine::serializeWithConfig]
    pub fn serialize_with_config(
        &'_ self,
        config: &mut SerializationConfig,
    ) -> Result<HostMemory<'engine>> {
        if !cfg!(feature = "mock_runtime") {
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

    /// See [`nvinfer1::ICudaEngine::isShapeInferenceIO`].
    pub fn is_shape_inference_io(&self, name: &str) -> Result<bool> {
        let name_cstr = std::ffi::CString::new(name)?;
        Ok(unsafe { self.inner.isShapeInferenceIO(name_cstr.as_ptr()) })
    }
}

#[cfg(test)]
#[cfg(not(feature = "mock_runtime"))]
mod tests {
    use crate::builder::network_flags;
    use crate::builder::{Builder, MemoryPoolType};
    use crate::logger::Logger;
    use crate::runtime::Runtime;
    use crate::{CudaEngine, DataType};
    use trtx_sys::{ActivationType, LayerInformationFormat};

    /// Build a minimal serialized engine with ProfilingVerbosity::kVERBOSE so inspector has layer info.
    fn build_minimal_engine_with_verbose_profiling(logger: &Logger) -> crate::Result<Vec<u8>> {
        let mut builder = Builder::new(logger)?;
        let mut network = builder.create_network(network_flags::EXPLICIT_BATCH)?;
        let mut tensor = network.add_input("input", DataType::kFLOAT, &[1, 4])?;
        tensor = network
            .add_activation(&tensor, ActivationType::kRELU)
            .unwrap()
            .output(&network, 0)
            .unwrap();
        tensor = network
            .add_activation(&tensor, ActivationType::kRELU)
            .unwrap()
            .output(&network, 0)
            .unwrap();
        network.mark_output(&tensor);

        let mut config = builder.create_config()?;
        config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, 1 << 20);
        config.set_profiling_verbosity(crate::ProfilingVerbosity::kDETAILED);

        let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
        Ok(engine_data.to_vec())
    }

    #[test]
    fn engine_inspector_json_verbose_profiling() {
        let logger = Logger::stderr().expect("logger");
        let engine_data =
            build_minimal_engine_with_verbose_profiling(&logger).expect("build engine");

        let mut runtime = Runtime::new(&logger).expect("runtime");
        let engine: CudaEngine<'_> = runtime
            .deserialize_cuda_engine(&engine_data)
            .expect("deserialize");

        let inspector = engine.create_engine_inspector().expect("engine inspector");
        let json = inspector
            .engine_information(LayerInformationFormat::kJSON)
            .expect("get_engine_information JSON");

        assert!(
            !json.is_empty(),
            "engine information JSON should not be empty"
        );
        assert!(
            json.trim_start().starts_with('{'),
            "engine information should be JSON (starts with '{{'); got: {}...",
            json.chars().take(80).collect::<String>()
        );
    }
}
