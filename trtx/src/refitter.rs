use std::marker::PhantomData;

use crate::{
    error::{Error, Result},
    CudaEngine, Logger,
};
use autocxx::cxx::UniquePtr;
use trtx_sys::nvinfer1;

pub struct Refitter<'logger, 'engine> {
    inner: UniquePtr<nvinfer1::IRefitter>,
    _logger: PhantomData<&'logger Logger>,
    _engine: PhantomData<&'engine CudaEngine<'engine>>,
}

impl<'logger, 'engine> Refitter<'logger, 'engine> {
    #[cfg(not(feature = "link_tensorrt_rtx"))]
    #[cfg(not(feature = "dlopen_tensorrt_rtx"))]
    pub fn new(logger: &'a Logger) -> Result<Self> {
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
                _engine: Default::default(),
                _logger: Default::default(),
            })
        }
        #[cfg(feature = "mock")]
        Ok(Builder {
            inner: UniquePtr::null(),
            _logger: Default::default(),
        })
    }
}
