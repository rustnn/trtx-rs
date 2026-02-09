//! CUDA memory management utilities
//!
//! Delegates to real/ or mock/ based on feature flag.

#[cfg(not(feature = "mock"))]
pub use crate::real::cuda::*;
#[cfg(feature = "mock")]
pub use crate::mock::cuda::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_buffer_allocation() {
        let buffer = DeviceBuffer::new(1024);
        assert!(buffer.is_ok());
        let buffer = buffer.unwrap();
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_device_buffer_copy() {
        let mut buffer = DeviceBuffer::new(256).unwrap();
        let host_data = vec![42u8; 256];
        assert!(buffer.copy_from_host(&host_data).is_ok());
        let mut output = vec![0u8; 256];
        assert!(buffer.copy_to_host(&mut output).is_ok());
        assert_eq!(host_data, output);
    }

    #[test]
    fn test_synchronize() {
        assert!(synchronize().is_ok());
    }
}
