//! Builder for creating TensorRT engines
//!
//! Delegates to real/ or mock/ based on feature flag.

/// Network definition builder flags
pub mod network_flags {
    /// Explicit batch sizes
    pub const EXPLICIT_BATCH: u32 = 1 << 0;
}

pub use trtx_sys::nvinfer1::MemoryPoolType;

#[cfg(feature = "mock")]
pub use crate::mock::builder::{Builder, BuilderConfig};
#[cfg(not(feature = "mock"))]
pub use crate::real::builder::{Builder, BuilderConfig};
