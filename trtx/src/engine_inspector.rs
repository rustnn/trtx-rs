use crate::{Error, Result};
use std::{ffi::CStr, marker::PhantomData};

use autocxx::cxx::UniquePtr;
use trtx_sys::{nvinfer1, LayerInformationFormat};

/// Engine inspector for layer/engine information (real mode).
/// See [`trtx_sys::nvinfer1::IEngineInspector`].
pub struct EngineInspector<'engine> {
    pub(crate) inner: UniquePtr<nvinfer1::IEngineInspector>,
    pub(crate) _engine: PhantomData<&'engine nvinfer1::ICudaEngine>,
}

impl EngineInspector<'_> {
    /// Returns layer information for the given layer index in the requested format.
    /// See [`trtx_sys::nvinfer1::IEngineInspector::getLayerInformation`].
    pub fn get_layer_information(
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

    /// Returns engine information in the requested format.
    /// See [`trtx_sys::nvinfer1::IEngineInspector::getEngineInformation`].
    pub fn get_engine_information(&self, format: LayerInformationFormat) -> Result<String> {
        let ptr = self.inner.getEngineInformation(format.into());
        Ok(if ptr.is_null() {
            return Err(Error::Runtime(
                "Could not get layer information".to_string(),
            ));
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_str()?.to_string()
        })
    }
}
