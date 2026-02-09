//! Builder for creating TensorRT engines
//!
//! Delegates to real/ or mock/ based on feature flag.

/// Network definition builder flags
pub mod network_flags {
    /// Explicit batch sizes
    pub const EXPLICIT_BATCH: u32 = 1 << 0;
}

/// Memory pool types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MemoryPoolType {
    /// Workspace memory
    Workspace = 0,
    /// DLA managed SRAM
    DlaManagedSram = 1,
    /// DLA local DRAM
    DlaLocalDram = 2,
    /// DLA global DRAM
    DlaGlobalDram = 3,
}

#[cfg(not(feature = "mock"))]
pub use crate::real::builder::{Builder, BuilderConfig};
#[cfg(feature = "mock")]
pub use crate::mock::builder::{Builder, BuilderConfig};
