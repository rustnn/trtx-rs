//! Optimization profile for dynamic input shapes (min / opt / max dimensions).
//!
//! [`OptimizationProfile`] wraps [`trtx_sys::nvinfer1::IOptimizationProfile`] (C++ [`nvinfer1::IOptimizationProfile`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1_i_optimization_profile.html)).

use std::{ffi::CString, marker::PhantomData, pin::Pin};

use crate::{error::PropertySetAttempt, Error, Result};
use trtx_sys::{nvinfer1, Dims64, OptProfileSelector};

/// [nvinfer1::IOptimizationProfile] — C++ [`nvinfer1::IOptimizationProfile`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1_i_optimization_profile.html).
pub struct OptimizationProfile<'builder> {
    pub(crate) inner: Pin<&'builder mut nvinfer1::IOptimizationProfile>,
    _builder: PhantomData<&'builder nvinfer1::IBuilder>,
}

impl std::fmt::Debug for OptimizationProfile<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptimizationProfile")
            .field("inner", &format!("{:x}", &*self.inner as *const _ as usize))
            .finish_non_exhaustive()
    }
}

impl<'builder> OptimizationProfile<'builder> {
    pub fn from_raw(profile: &'builder mut nvinfer1::IOptimizationProfile) -> Self {
        Self {
            inner: unsafe { Pin::new_unchecked(profile) },
            _builder: Default::default(),
        }
    }

    /// Overrides the builder config's profiling stream for this profile.
    /// Pass null to clear the override.
    ///
    /// # Safety
    /// The CUDA stream and any owning green context must remain valid until the build completes.
    #[cfg(all(feature = "v_1_7", not(feature = "enterprise")))]
    pub unsafe fn set_profile_stream(&mut self, stream: *mut std::ffi::c_void) {
        if !cfg!(feature = "mock_runtime") {
            self.inner.as_mut().setProfileStream(stream.cast());
        }
    }

    /// Returns the profile's profiling stream override, or null when none is set.
    #[cfg(all(feature = "v_1_7", not(feature = "enterprise")))]
    pub fn profile_stream(&self) -> *mut std::ffi::c_void {
        if cfg!(feature = "mock_runtime") {
            std::ptr::null_mut()
        } else {
            self.inner.getProfileStream().cast()
        }
    }

    /// See [nvinfer1::IOptimizationProfile::getDimensions]
    pub fn dimensions(&self, input_name: &str, select: OptProfileSelector) -> Dims64 {
        let input_name_c =
            CString::new(input_name).expect("User provided string that contains \\0 characters");
        unsafe {
            self.inner
                .getDimensions(input_name_c.as_ptr(), select.into())
        }
    }

    /// See [nvinfer1::IOptimizationProfile::setDimensions]
    pub fn set_dimensions(
        &mut self,
        input_name: &str,
        select: OptProfileSelector,
        dims: &Dims64,
    ) -> Result<()> {
        let input_name_c =
            CString::new(input_name).expect("User provided string that contains \\0 characters");
        unsafe {
            if self
                .inner
                .as_mut()
                .setDimensions(input_name_c.as_ptr(), select.into(), dims)
            {
                Ok(())
            } else {
                Err(Error::FailedToSetProperty(
                    PropertySetAttempt::OptimizationProfileSetDimensions,
                ))
            }
        }
    }

    /// See [nvinfer1::IOptimizationProfile::isValid]
    pub fn is_valid(&self) -> bool {
        self.inner.isValid()
    }

    /// See [nvinfer1::IOptimizationProfile::setExtraMemoryTarget]
    pub fn set_extra_memory_target(&mut self, target: f32) -> Result<()> {
        if self.inner.as_mut().setExtraMemoryTarget(target) {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(
                PropertySetAttempt::OptimizationProfileSetExtraMemoryTarget,
            ))
        }
    }

    /// See [nvinfer1::IOptimizationProfile::getExtraMemoryTarget]
    pub fn extra_memory_target(&self) -> f32 {
        self.inner.getExtraMemoryTarget()
    }

    /// See [nvinfer1::IOptimizationProfile::setShapeValuesV2]
    pub fn set_shape_values_v2(
        &mut self,
        input_name: &str,
        select: OptProfileSelector,
        values: &[i64],
    ) -> Result<()> {
        let input_name_c =
            CString::new(input_name).expect("User provided string that contains \\0 characters");
        if unsafe {
            self.inner.as_mut().setShapeValuesV2(
                input_name_c.as_ptr(),
                select.into(),
                values.as_ptr(),
                values.len().try_into().expect("Vector to long for a i32"),
            )
        } {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(
                PropertySetAttempt::OptimizationProfileSetShapeValues,
            ))
        }
    }
}

impl Drop for OptimizationProfile<'_> {
    fn drop(&mut self) {
        unsafe {
            std::ptr::drop_in_place(self.inner.as_mut().get_unchecked_mut());
        }
    }
}
