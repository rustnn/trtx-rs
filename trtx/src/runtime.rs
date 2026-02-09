//! Runtime for deserializing and managing TensorRT engines
//!
//! Delegates to real/ or mock/ based on feature flag.

#[cfg(not(feature = "mock"))]
pub use crate::real::runtime::*;
#[cfg(feature = "mock")]
pub use crate::mock::runtime::*;
