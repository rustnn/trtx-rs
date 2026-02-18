use std::{ffi::CString, marker::PhantomData, pin::Pin};

use crate::{error::PropertySetAttempt, Error, Result};
use trtx_sys::{nvinfer1, Dims64, OptProfileSelector};

/// See [nvinfer1::IOptimizationProfile]
pub struct OptimizationProfile<'builder> {
    pub(crate) inner: Pin<&'builder mut nvinfer1::IOptimizationProfile>,
    _builder: PhantomData<&'builder nvinfer1::IBuilder>,
}

impl<'builder> OptimizationProfile<'builder> {
    pub fn from_raw(profile: &'builder mut nvinfer1::IOptimizationProfile) -> Self {
        Self {
            inner: unsafe { Pin::new_unchecked(profile) },
            _builder: Default::default(),
        }
    }

    pub fn get_dimensions(&self, input_name: &str, select: OptProfileSelector) -> Dims64 {
        let input_name_c =
            CString::new(input_name).expect("User provided string that contains \\0 characters");
        unsafe {
            self.inner
                .getDimensions(input_name_c.as_ptr(), select.into())
        }
    }
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

    pub fn is_valid(&self) -> bool {
        self.inner.isValid()
    }

    pub fn set_extra_memory_target(&mut self, target: f32) -> Result<()> {
        if self.inner.as_mut().setExtraMemoryTarget(target) {
            Ok(())
        } else {
            Err(Error::FailedToSetProperty(
                PropertySetAttempt::OptimizationProfileSetExtraMemoryTarget,
            ))
        }
    }
    pub fn get_extra_memory_target(&self) -> f32 {
        self.inner.getExtraMemoryTarget()
    }

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
