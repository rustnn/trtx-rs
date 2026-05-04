//! Engine inspector (layer / engine introspection as text or JSON).
//!
//! [`EngineInspector`] wraps [`trtx_sys::nvinfer1::IEngineInspector`] (C++ [`nvinfer1::IEngineInspector`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_engine_inspector.html)).

use crate::{Error, Result};
use std::{ffi::CStr, marker::PhantomData};

use autocxx::cxx::UniquePtr;
use trtx_sys::{nvinfer1, LayerInformationFormat};

/// [`trtx_sys::nvinfer1::IEngineInspector`] — C++ [`nvinfer1::IEngineInspector`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_engine_inspector.html).
pub struct EngineInspector<'engine> {
    pub(crate) inner: UniquePtr<nvinfer1::IEngineInspector>,
    pub(crate) _engine: PhantomData<&'engine nvinfer1::ICudaEngine>,
}

impl std::fmt::Debug for EngineInspector<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineInspector")
            .field("inner", &format!("{:x}", self.inner.as_ptr() as usize))
            .finish_non_exhaustive()
    }
}

impl EngineInspector<'_> {
    /// Returns layer information for the given layer index in the requested format.
    /// See [`trtx_sys::nvinfer1::IEngineInspector::getLayerInformation`].
    pub fn layer_information(
        &mut self,
        layer_index: i32,
        format: LayerInformationFormat,
    ) -> Result<String> {
        let ptr = self.inner.getLayerInformation(layer_index, format.into());
        Ok(if ptr.is_null() {
            return Err(Error::Runtime(
                "Could not get layer information".to_string(),
            ));
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string()
        })
    }

    #[deprecated = "use layer_information instead"]
    pub fn get_layer_information(
        &mut self,
        layer_index: i32,
        format: LayerInformationFormat,
    ) -> Result<String> {
        self.layer_information(layer_index, format)
    }

    /// Returns engine information in the requested format.
    /// See [`trtx_sys::nvinfer1::IEngineInspector::getEngineInformation`].
    pub fn engine_information(&self, format: LayerInformationFormat) -> Result<String> {
        let ptr = self.inner.getEngineInformation(format.into());
        Ok(if ptr.is_null() {
            return Err(Error::Runtime(
                "Could not get layer information".to_string(),
            ));
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string()
        })
    }

    #[deprecated = "use engine_information instead"]
    pub fn get_engine_information(&self, format: LayerInformationFormat) -> Result<String> {
        self.engine_information(format)
    }
}
