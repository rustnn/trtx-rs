//! Executor module providing rustnn-compatible interface
//!
//! This module provides a simplified API for executing ONNX models with TensorRT,
//! designed to integrate easily with rustnn's executor pattern.

#[cfg(feature = "onnxparser")]
use crate::builder::network_flags;
use crate::cuda::DeviceBuffer;
use crate::error::Result;
#[cfg(feature = "onnxparser")]
use crate::{Builder, OnnxParser};
use crate::{Logger, Runtime};

/// Input descriptor for TensorRT execution
#[derive(Debug, Clone)]
pub struct TensorInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Output descriptor from TensorRT execution
#[derive(Debug, Clone)]
pub struct TensorOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Execute an ONNX model with TensorRT using provided inputs
///
/// This function follows the rustnn executor pattern:
/// 1. Parse ONNX model
/// 2. Build TensorRT engine
/// 3. Execute inference
/// 4. Return results
///
/// # Arguments
///
/// * `onnx_model_bytes` - ONNX model as byte slice
/// * `inputs` - Input tensors with names, shapes, and data
///
/// # Returns
///
/// Vector of output tensors with names, shapes, and computed data
#[cfg(feature = "onnxparser")]
pub fn run_onnx_with_tensorrt(
    onnx_model_bytes: &[u8],
    inputs: &[TensorInput],
) -> Result<Vec<TensorOutput>> {
    // Create logger
    let logger = Logger::stderr()?;

    // Build engine from ONNX
    let engine_data = build_engine_from_onnx(&logger, onnx_model_bytes)?;

    // Execute inference
    execute_engine(&logger, &engine_data, inputs)
}

/// Build TensorRT engine from ONNX model
#[cfg(feature = "onnxparser")]
fn build_engine_from_onnx(logger: &Logger, onnx_bytes: &[u8]) -> Result<Vec<u8>> {
    // Create builder
    let builder = Builder::new(logger)?;

    // Create network with explicit batch
    let mut network = builder.create_network(network_flags::EXPLICIT_BATCH)?;

    // Parse ONNX model
    let parser = OnnxParser::new(&mut network, logger)?;
    parser.parse(onnx_bytes)?;

    // Configure builder
    let mut config = builder.create_config()?;

    // Set workspace memory (1GB)
    config.set_memory_pool_limit(crate::builder::MemoryPoolType::Workspace, 1 << 30)?;

    // Build serialized engine
    builder.build_serialized_network(&mut network, &mut config)
}

/// Execute TensorRT engine with inputs
fn execute_engine(
    logger: &Logger,
    engine_data: &[u8],
    inputs: &[TensorInput],
) -> Result<Vec<TensorOutput>> {
    // Create runtime and deserialize engine
    let runtime = Runtime::new(logger)?;
    let engine = runtime.deserialize_cuda_engine(engine_data)?;
    let mut context = engine.create_execution_context()?;

    // Get tensor information
    let num_tensors = engine.get_nb_io_tensors()?;

    // Prepare CUDA buffers for inputs and outputs
    let mut device_buffers: Vec<(String, DeviceBuffer)> = Vec::new();
    let mut output_info: Vec<(String, Vec<usize>)> = Vec::new();

    // Process each tensor
    for i in 0..num_tensors {
        let name = engine.get_tensor_name(i)?;

        // Check if this is an input or output
        if let Some(input) = inputs.iter().find(|inp| inp.name == name) {
            // Input tensor - validate shape matches engine expectations
            let expected_shape_i64 = engine.get_tensor_shape(&name)?;
            let expected_shape: Vec<usize> =
                expected_shape_i64.iter().map(|&d| d as usize).collect();
            let expected_elements: usize = expected_shape.iter().product();
            let provided_elements: usize = input.shape.iter().product();

            if provided_elements != expected_elements {
                return Err(crate::Error::InvalidArgument(format!(
                    "Input tensor '{}' shape mismatch: expected {:?} ({} elements), got {:?} ({} elements)",
                    name, expected_shape, expected_elements, input.shape, provided_elements
                )));
            }

            // Validate data length matches shape
            if input.data.len() != provided_elements {
                return Err(crate::Error::InvalidArgument(format!(
                    "Input tensor '{}' data length ({}) doesn't match shape {:?} ({} elements)",
                    name,
                    input.data.len(),
                    input.shape,
                    provided_elements
                )));
            }

            // Allocate and copy data
            let size_bytes = input.data.len() * std::mem::size_of::<f32>();
            let mut buffer = DeviceBuffer::new(size_bytes)?;

            // Copy input data to device
            let input_bytes =
                unsafe { std::slice::from_raw_parts(input.data.as_ptr() as *const u8, size_bytes) };
            buffer.copy_from_host(input_bytes)?;

            // Bind tensor address
            unsafe {
                context.set_tensor_address(&name, buffer.as_ptr())?;
            }

            device_buffers.push((name.clone(), buffer));
        } else {
            // Output tensor - query actual shape from engine
            let shape_i64 = engine.get_tensor_shape(&name)?;
            let shape: Vec<usize> = shape_i64.iter().map(|&d| d as usize).collect();

            // Calculate actual buffer size needed
            let num_elements: usize = shape.iter().product();
            let size_bytes = num_elements * std::mem::size_of::<f32>();
            let buffer = DeviceBuffer::new(size_bytes)?;

            unsafe {
                context.set_tensor_address(&name, buffer.as_ptr())?;
            }

            output_info.push((name.clone(), shape));
            device_buffers.push((name.clone(), buffer));
        }
    }

    // Execute inference
    unsafe {
        context.enqueue_v3(crate::cuda::get_default_stream())?;
    }

    // Synchronize to ensure completion
    crate::cuda::synchronize()?;

    // Copy outputs back to host
    let mut outputs = Vec::new();

    for (name, shape) in output_info {
        if let Some((_, buffer)) = device_buffers.iter().find(|(n, _)| n == &name) {
            let size_bytes = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
            let mut host_data: Vec<u8> = vec![0u8; size_bytes];

            buffer.copy_to_host(host_data.as_mut_slice())?;

            // Convert bytes to f32
            let data: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    host_data.as_ptr() as *const f32,
                    size_bytes / std::mem::size_of::<f32>(),
                )
            }
            .to_vec();

            outputs.push(TensorOutput { name, shape, data });
        }
    }

    Ok(outputs)
}

/// Simpler version: Execute with zero-filled inputs (useful for testing/validation)
#[cfg(feature = "onnxparser")]
pub fn run_onnx_zeroed(
    onnx_model_bytes: &[u8],
    input_descriptors: &[(String, Vec<usize>)],
) -> Result<Vec<TensorOutput>> {
    // Create zero-filled inputs
    let inputs: Vec<TensorInput> = input_descriptors
        .iter()
        .map(|(name, shape)| {
            let size: usize = shape.iter().product();
            TensorInput {
                name: name.clone(),
                shape: shape.clone(),
                data: vec![0.0; size],
            }
        })
        .collect();

    run_onnx_with_tensorrt(onnx_model_bytes, &inputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_input_creation() {
        let input = TensorInput {
            name: "input".to_string(),
            shape: vec![1, 3, 224, 224],
            data: vec![0.0; 3 * 224 * 224],
        };

        assert_eq!(input.name, "input");
        assert_eq!(input.shape, vec![1, 3, 224, 224]);
        assert_eq!(input.data.len(), 3 * 224 * 224);
    }

    #[test]
    #[ignore] // Requires valid ONNX model
    #[cfg(feature = "onnxparser")]
    fn test_executor_basic() {
        let dummy_onnx = vec![0u8; 100];
        let inputs = vec![("input".to_string(), vec![1, 3, 224, 224])];

        let _result = run_onnx_zeroed(&dummy_onnx, &inputs);
        // In mock mode, this should succeed
        #[cfg(feature = "mock")]
        assert!(_result.is_ok());
    }
}
