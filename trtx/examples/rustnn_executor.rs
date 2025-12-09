//! Example demonstrating rustnn-compatible executor API
//!
//! This example shows how to use trtx as an executor in the rustnn pattern:
//! - Load ONNX model bytes
//! - Execute with TensorRT
//! - Get output tensors
//!
//! Run with: cargo run --features mock --example rustnn_executor

use std::error::Error;
use trtx::executor::{run_onnx_with_tensorrt, run_onnx_zeroed, TensorInput};

fn main() -> Result<(), Box<dyn Error>> {
    println!("TensorRT-RTX Executor for rustnn");
    println!("==================================\n");

    // Example 1: Execute with zero-filled inputs (for testing)
    println!("1. Testing with zero-filled inputs...");

    let dummy_onnx = create_dummy_onnx_model();

    let input_descriptors = vec![("input".to_string(), vec![1, 3, 224, 224])];

    match run_onnx_zeroed(&dummy_onnx, &input_descriptors) {
        Ok(outputs) => {
            println!("   ✓ Execution succeeded");
            println!("   Outputs:");
            for output in outputs {
                println!(
                    "      - {}: shape {:?}, {} values",
                    output.name,
                    output.shape,
                    output.data.len()
                );
            }
        }
        Err(e) => {
            println!("   ✗ Execution failed: {}", e);
            println!("   (This is expected with a dummy model in mock mode)");
        }
    }

    // Example 2: Execute with actual input data
    println!("\n2. Testing with actual input data...");

    let inputs = vec![TensorInput {
        name: "input".to_string(),
        shape: vec![1, 3, 224, 224],
        data: create_sample_input(3 * 224 * 224),
    }];

    match run_onnx_with_tensorrt(&dummy_onnx, &inputs) {
        Ok(outputs) => {
            println!("   ✓ Execution succeeded");
            for output in outputs {
                println!("      - {}: shape {:?}", output.name, output.shape);
                println!(
                    "        First 5 values: {:?}",
                    &output.data[..output.data.len().min(5)]
                );
            }
        }
        Err(e) => {
            println!("   ✗ Execution failed: {}", e);
            println!("   (Expected with dummy model - use real ONNX for actual inference)");
        }
    }

    println!("\n3. rustnn Integration Pattern");
    println!("   To use in rustnn, implement:");
    println!("   ```rust");
    println!("   #[cfg(feature = \"trtx-runtime\")]");
    println!("   pub fn run_trtx_with_inputs(");
    println!("       model_bytes: &[u8],");
    println!("       inputs: &[TrtxInput],");
    println!("   ) -> Result<Vec<TrtxOutputWithData>> {{");
    println!("       trtx::run_onnx_with_tensorrt(model_bytes, inputs)");
    println!("   }}");
    println!("   ```");

    println!("\n✓ Example completed");

    Ok(())
}

/// Create a dummy ONNX model for demonstration
/// In real usage, you would load an actual ONNX file
fn create_dummy_onnx_model() -> Vec<u8> {
    // This is just a placeholder - in real usage, load from file:
    // std::fs::read("model.onnx")?
    vec![0u8; 100]
}

/// Create sample input data (random-ish values)
fn create_sample_input(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * 0.001).sin()).collect()
}
