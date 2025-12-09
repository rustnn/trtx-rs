//! Executor module providing rustnn-compatible interface
//!
//! This module provides a simplified API for executing ONNX models with TensorRT,
//! designed to integrate easily with rustnn's executor pattern.

use crate::builder::network_flags;
use crate::cuda::DeviceBuffer;
use crate::error::Result;
use crate::{Builder, Logger, OnnxParser, Runtime};

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
fn build_engine_from_onnx(logger: &Logger, onnx_bytes: &[u8]) -> Result<Vec<u8>> {
    // Create builder
    let builder = Builder::new(logger)?;

    // Create network with explicit batch
    let network = builder.create_network(network_flags::EXPLICIT_BATCH)?;

    // Parse ONNX model
    let parser = OnnxParser::new(&network, logger)?;
    parser.parse(onnx_bytes)?;

    // Configure builder
    let mut config = builder.create_config()?;

    // Set workspace memory (1GB)
    config.set_memory_pool_limit(crate::builder::MemoryPoolType::Workspace, 1 << 30)?;

    // Build serialized engine
    builder.build_serialized_network(&network, &config)
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
            // Input tensor - allocate and copy data
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
            // Output tensor - allocate buffer
            // Note: In a real implementation, we would query the tensor shape
            // For now, we'll use a reasonable default size
            let estimated_size = 1000 * std::mem::size_of::<f32>();
            let buffer = DeviceBuffer::new(estimated_size)?;

            unsafe {
                context.set_tensor_address(&name, buffer.as_ptr())?;
            }

            output_info.push((name.clone(), vec![1, 1000])); // Dummy shape
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
            let mut host_data = vec![0u8; size_bytes];

            buffer.copy_to_host(&mut host_data)?;

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
    fn test_executor_basic() {
        let dummy_onnx = vec![0u8; 100];
        let inputs = vec![("input".to_string(), vec![1, 3, 224, 224])];

        let result = run_onnx_zeroed(&dummy_onnx, &inputs);
        // In mock mode, this should succeed
        #[cfg(feature = "mock")]
        assert!(result.is_ok());
    }
}
