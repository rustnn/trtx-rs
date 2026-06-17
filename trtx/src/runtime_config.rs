//! Runtime configuration for execution context creation.
//!
//! [`RuntimeConfig`] wraps [`trtx_sys::nvinfer1::IRuntimeConfig`] (C++ [`nvinfer1::IRuntimeConfig`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_runtime_config.html)).

use std::marker::PhantomData;
#[cfg(not(feature = "enterprise"))]
use std::sync::{Arc, Mutex};

#[cfg(not(feature = "enterprise"))]
use crate::error::PropertySetAttempt;
use crate::error::Result;
#[cfg(not(feature = "enterprise"))]
use crate::runtime_cache::RuntimeCache;
use crate::Error;
use cxx::UniquePtr;
use trtx_sys::nvinfer1::{self, IRuntimeConfig};
use trtx_sys::ExecutionContextAllocationStrategy;

#[cfg(not(feature = "enterprise"))]
use trtx_sys::{CudaGraphStrategy, DynamicShapesKernelSpecializationStrategy};

/// [`trtx_sys::nvinfer1::IRuntimeConfig`] — C++ [`nvinfer1::IRuntimeConfig`](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/classnvinfer1_1_1_i_runtime_config.html).
pub struct RuntimeConfig<'engine> {
    pub(crate) inner: UniquePtr<IRuntimeConfig>,
    _engine: PhantomData<&'engine nvinfer1::ICudaEngine>,
    // actually IRuntimeCache has its mutex, so we could omit this if we made mut methods of RuntimeCache (e.g. deserialize &self)
    // this also makes it safe when we modify through our mutex, while cpp calls are made through
    // IExecution calls
    #[cfg(not(feature = "enterprise"))]
    _cache: Option<Arc<Mutex<RuntimeCache<'engine>>>>, // Mutex, could now be removed with a
                                                       // breaking change to set_runtime_cache
}

impl std::fmt::Debug for RuntimeConfig<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeConfig")
            .field("inner", &format!("{:x}", self.inner.as_ptr() as usize))
            .finish_non_exhaustive()
    }
}

/// # Safety
///
/// Transferring to other thread is safe, as
/// - it is safe for IRuntimeConfig from C++ API
/// - UniquePtr always holds a valid IBuilder and is only mutated in initializer
/// - RuntimeCache is Send+Sync
unsafe impl Send for RuntimeConfig<'_> {}

impl<'engine> RuntimeConfig<'engine> {
    pub(crate) fn new(runtime_config: *mut nvinfer1::IRuntimeConfig) -> Result<Self> {
        #[cfg(not(feature = "mock"))]
        if runtime_config.is_null() {
            return Err(Error::RuntimeConfigCreationFailed);
        }
        Ok(Self {
            inner: unsafe { UniquePtr::from_raw(runtime_config) },
            _engine: Default::default(),
            #[cfg(not(feature = "enterprise"))]
            _cache: None,
        })
    }

    /// See [IRuntimeConfig::setExecutionContextAllocationStrategy].
    pub fn set_execution_context_allocation_strategy(
        &mut self,
        strategy: ExecutionContextAllocationStrategy,
    ) {
        #[cfg(not(feature = "mock"))]
        self.inner
            .pin_mut()
            .setExecutionContextAllocationStrategy(strategy.into());
    }

    /// See [IRuntimeConfig::getExecutionContextAllocationStrategy].
    pub fn execution_context_allocation_strategy(&self) -> ExecutionContextAllocationStrategy {
        if cfg!(not(feature = "mock")) {
            self.inner.getExecutionContextAllocationStrategy().into()
        } else {
            ExecutionContextAllocationStrategy::kSTATIC
        }
    }

    #[cfg(not(feature = "enterprise"))]
    /// See [IRuntimeConfig::createRuntimeCache].
    pub fn create_runtime_cache(&self) -> Result<RuntimeCache<'engine>> {
        #[cfg(not(feature = "mock"))]
        let cache_ptr = self.inner.createRuntimeCache();
        #[cfg(feature = "mock")]
        let cache_ptr = std::ptr::null_mut();
        RuntimeCache::new(cache_ptr)
    }

    #[cfg(not(feature = "enterprise"))]
    /// See [IRuntimeConfig::setRuntimeCache].
    pub fn set_runtime_cache(&mut self, cache: Arc<Mutex<RuntimeCache<'engine>>>) -> Result<()> {
        if cfg!(not(feature = "mock")) {
            if self.inner.pin_mut().setRuntimeCache(
                cache
                    .lock()
                    .unwrap()
                    .inner
                    .as_ref()
                    .expect("RuntimeCache inner must be non-null"),
            ) {
                self._cache = Some(cache);
                Ok(())
            } else {
                Err(Error::FailedToSetProperty(
                    PropertySetAttempt::RuntimeConfigRuntimeCache,
                ))
            }
        } else {
            Ok(())
        }
    }

    #[cfg(not(feature = "enterprise"))]
    /// See [IRuntimeConfig::getRuntimeCache].
    ///
    /// Returns `None` if no runtime cache has been set.
    pub fn runtime_cache(&self) -> Option<*mut nvinfer1::IRuntimeCache> {
        if cfg!(not(feature = "mock")) {
            let ptr = self.inner.getRuntimeCache();
            if ptr.is_null() {
                None
            } else {
                Some(ptr)
            }
        } else {
            None
        }
    }

    #[cfg(not(feature = "enterprise"))]
    /// See [IRuntimeConfig::setDynamicShapesKernelSpecializationStrategy].
    pub fn set_dynamic_shapes_kernel_specialization_strategy(
        &mut self,
        strategy: DynamicShapesKernelSpecializationStrategy,
    ) {
        #[cfg(not(feature = "mock"))]
        self.inner
            .pin_mut()
            .setDynamicShapesKernelSpecializationStrategy(strategy.into());
    }

    #[cfg(not(feature = "enterprise"))]
    /// See [IRuntimeConfig::getDynamicShapesKernelSpecializationStrategy].
    pub fn dynamic_shapes_kernel_specialization_strategy(
        &self,
    ) -> DynamicShapesKernelSpecializationStrategy {
        if cfg!(not(feature = "mock")) {
            self.inner
                .getDynamicShapesKernelSpecializationStrategy()
                .into()
        } else {
            DynamicShapesKernelSpecializationStrategy::kNONE
        }
    }

    #[cfg(not(feature = "enterprise"))]
    /// See [IRuntimeConfig::setCudaGraphStrategy].
    pub fn set_cuda_graph_strategy(&mut self, strategy: CudaGraphStrategy) -> Result<()> {
        if cfg!(not(feature = "mock")) {
            if self.inner.pin_mut().setCudaGraphStrategy(strategy.into()) {
                Ok(())
            } else {
                Err(Error::FailedToSetProperty(
                    PropertySetAttempt::RuntimeConfigCudaGraphStrategy,
                ))
            }
        } else {
            Ok(())
        }
    }

    #[cfg(not(feature = "enterprise"))]
    /// See [IRuntimeConfig::getCudaGraphStrategy].
    pub fn cuda_graph_strategy(&self) -> CudaGraphStrategy {
        if cfg!(not(feature = "mock")) {
            self.inner.getCudaGraphStrategy().into()
        } else {
            CudaGraphStrategy::kDISABLED
        }
    }
}
