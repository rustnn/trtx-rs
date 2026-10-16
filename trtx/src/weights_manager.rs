//! Save and restore the opaque weight data of a weight-stripped engine.

use std::{ffi::c_void, marker::PhantomData, pin::Pin};

use cxx::UniquePtr;
use trtx_sys::{nvinfer1, TensorLocation};

use crate::{Error, Result};

/// Owns an [`nvinfer1::IWeightsManager`] and borrows its engine until dropped.
///
/// Create with [`crate::CudaEngine::create_weights_manager`]. Saved data is opaque:
/// preserve every byte, including allocation padding. Restore it only to the engine
/// that produced it or an engine deserialized from the same plan with a compatible
/// runtime and CUDA device.
///
/// Weight memory changes must not overlap inference, refit, serialization, execution
/// context creation, weight-streaming changes, or another manager operation for the
/// same engine. A mutable borrow of this wrapper alone cannot enforce those rules.
pub struct WeightsManager<'engine> {
    inner: UniquePtr<nvinfer1::IWeightsManager>,
    _engine: PhantomData<&'engine nvinfer1::ICudaEngine>,
}

impl std::fmt::Debug for WeightsManager<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightsManager")
            .field("inner", &self.inner.as_ptr())
            .finish_non_exhaustive()
    }
}

impl<'engine> WeightsManager<'engine> {
    /// Takes ownership of a manager whose engine is borrowed for `'engine`.
    pub(crate) unsafe fn from_raw(ptr: *mut nvinfer1::IWeightsManager) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                inner: unsafe { UniquePtr::from_raw(ptr) },
                _engine: PhantomData,
            })
        }
    }

    /// Number of bytes in a weight backup, including CUDA allocation padding.
    pub fn size(&self) -> i64 {
        self.inner.getSize()
    }

    fn buffer_size(&self, size: usize) -> Result<i64> {
        let size = i64::try_from(size)
            .map_err(|_| Error::InvalidArgument("Weight buffer is too large".into()))?;
        if size != self.size() {
            return Err(Error::InvalidArgument(
                "Weight buffer size must equal WeightsManager::size()".into(),
            ));
        }
        Ok(size)
    }

    /// Copies loaded weights to a host or device buffer, ordered on `stream`.
    ///
    /// # Safety
    /// `dst` must be writable for `size` bytes at `location` and remain valid and
    /// unused until the copy completes. `stream` must be valid for the engine's CUDA
    /// device and order prior engine work. The weights must be loaded; no conflicting
    /// engine or manager operations may overlap this transfer.
    pub unsafe fn save_to_buffer(
        &mut self,
        dst: *mut c_void,
        size: usize,
        location: TensorLocation,
        stream: *mut c_void,
    ) -> Result<()> {
        let size = self.buffer_size(size)?;
        let success = unsafe {
            self.inner
                .pin_mut()
                .saveToBuffer(dst.cast(), size, location.into(), stream.cast())
        };
        operation_result(success, "Failed to save engine weights to buffer")
    }

    /// Writes exactly [`Self::size`] bytes to a TensorRT stream writer.
    /// TensorRT synchronizes `stream` before passing each host chunk to the writer.
    /// Chunk data is valid only during the writer's `write` callback.
    ///
    /// # Safety
    /// `stream` must be valid for the engine's CUDA device and order prior engine
    /// work. The weights must be loaded. No conflicting engine or manager operations
    /// may overlap this call. The writer must obey `IStreamWriter`'s callback contract
    /// and must not unwind across the C++ boundary.
    pub unsafe fn save_to_stream(
        &mut self,
        writer: Pin<&mut nvinfer1::IStreamWriter>,
        stream: *mut c_void,
    ) -> Result<()> {
        let success = unsafe { self.inner.pin_mut().saveToStream(writer, stream.cast()) };
        operation_result(success, "Failed to save engine weights to stream")
    }

    /// Synchronizes `stream`, then unmaps weights while retaining their virtual addresses.
    /// Application VMM allocation handles remain the application's responsibility.
    ///
    /// # Safety
    /// `stream` must be valid for the engine's CUDA device and order all prior engine
    /// work. No conflicting engine or manager operations may overlap this call.
    /// Do not execute the engine again until all weights have been restored and the
    /// restore operations have completed.
    pub unsafe fn unload(&mut self, stream: *mut c_void) -> Result<()> {
        let success = unsafe { self.inner.pin_mut().unload(stream.cast()) };
        operation_result(success, "Failed to unload engine weights")
    }

    /// Allocates weight memory and restores an opaque backup without refitting.
    /// The copy is ordered on `stream` and may still be in flight on return.
    ///
    /// # Safety
    /// `src` must contain a compatible, unchanged backup of `size` bytes at `location`.
    /// Keep it alive and unchanged until the copy completes. `stream` must be valid
    /// for the engine's CUDA device. Weights must be unloaded or not yet loaded, and
    /// no conflicting engine or manager operations may overlap the restore. Order
    /// subsequent inference after the copy completes.
    pub unsafe fn restore_from_buffer(
        &mut self,
        src: *const c_void,
        size: usize,
        location: TensorLocation,
        stream: *mut c_void,
    ) -> Result<()> {
        let size = self.buffer_size(size)?;
        let success = unsafe {
            self.inner
                .pin_mut()
                .restoreFromBuffer(src.cast(), size, location.into(), stream.cast())
        };
        operation_result(success, "Failed to restore engine weights from buffer")
    }

    /// Maps an application CUDA VMM allocation into a range of the engine's weights.
    /// The allocation's physical offset zero supplies the backup range at `offset`.
    /// Multiple calls may restore nonoverlapping ranges.
    ///
    /// # Safety
    /// `allocation_handle` must be a live `cuMemCreate` allocation on the engine's CUDA
    /// device, contain the matching opaque backup bytes, and hold at least `size` bytes.
    /// Both `offset` and `size` must satisfy CUDA's minimum allocation granularity.
    /// Weights must be unloaded or partially initialized; the range must not overlap
    /// an existing mapping. `stream` must be valid for the engine's device and order
    /// prior work. No conflicting engine or manager operations may overlap this call.
    /// Inference must wait until the entire weight range has been restored and all
    /// restore operations have completed.
    ///
    /// Keep the handle alive until this call returns. TensorRT's successful mapping
    /// then retains the allocation; retain your handle too if you need to reuse the
    /// allocation after unloading or destroying the engine.
    pub unsafe fn restore_from_vmm_allocation(
        &mut self,
        allocation_handle: u64,
        offset: i64,
        size: i64,
        stream: *mut c_void,
    ) -> Result<()> {
        if offset < 0 || size <= 0 || offset.checked_add(size).is_none_or(|end| end > self.size()) {
            return Err(Error::InvalidArgument(
                "VMM weight range must lie within [0, WeightsManager::size())".into(),
            ));
        }
        let success = unsafe {
            self.inner.pin_mut().restoreFromVmmAllocation(
                allocation_handle,
                offset,
                size,
                stream.cast(),
            )
        };
        operation_result(
            success,
            "Failed to restore engine weights from VMM allocation",
        )
    }
}

fn operation_result(success: bool, message: &str) -> Result<()> {
    if success {
        Ok(())
    } else {
        Err(Error::Runtime(message.into()))
    }
}
