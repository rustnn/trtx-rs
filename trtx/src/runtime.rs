//! Runtime for deserializing and managing TensorRT engines
//!
//! Delegates to real/ or mock/ based on feature flag.

pub use crate::real::runtime::*;
