//! Basic TensorRT-RTX workflow example
//!
//! This example demonstrates:
//! 1. Creating a logger
//! 2. Building an engine
//! 3. Serializing to disk
//! 4. Deserializing and running inference
//!
//! Note: This is a skeleton example. Real usage requires:
//! - Adding layers to the network
//! - Allocating CUDA memory for tensors
//! - Copying data to/from GPU

use std::error::Error;
use trtx::builder::{network_flags, MemoryPoolType};
use trtx::{Builder, Logger, Runtime};

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "dlopen_tensorrt_rtx")]
    trtx::dynamically_load_tensorrt(None::<String>).unwrap();
    #[cfg(feature = "dlopen_tensorrt_onnxparser")]
    trtx::dynamically_load_tensorrt_onnxparser(None::<String>).unwrap();

    println!("TensorRT-RTX Basic Workflow Example");
    println!("=====================================\n");

    // Step 1: Create logger
    println!("1. Creating logger...");
    let logger = Logger::stderr()?;
    println!("   ✓ Logger created\n");

    // Step 2: Build phase
    println!("2. Building engine...");

    let builder = Builder::new(&logger)?;
    println!("   ✓ Builder created");

    // Create network with explicit batch dimensions
    let mut network = builder.create_network(network_flags::EXPLICIT_BATCH)?;
    println!("   ✓ Network created");

    // Create and configure builder config
    let mut config = builder.create_config()?;
    println!("   ✓ Config created");

    // Set workspace memory limit (1GB)
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 30)?;
    println!("   ✓ Workspace limit set to 1GB");

    // Note: In a real application, you would add layers to the network here
    // For example:
    // - network.add_input(...)
    // - network.add_convolution(...)
    // - network.add_activation(...)
    // - etc.

    println!("\n   Note: This example uses an empty network.");
    println!("   In production, you would:");
    println!("   - Parse an ONNX model");
    println!("   - Or programmatically add layers");
    println!("   - Define input/output tensors\n");

    // Build serialized network
    println!("   Building serialized engine...");
    match builder.build_serialized_network(&mut network, &mut config) {
        Ok(engine_data) => {
            println!("   ✓ Engine built ({} bytes)", engine_data.len());

            // Save to disk
            let engine_path = "/tmp/example.engine";
            std::fs::write(engine_path, &engine_data)?;
            println!("   ✓ Engine saved to {}\n", engine_path);

            // Step 3: Inference phase
            println!("3. Loading engine for inference...");

            let runtime = Runtime::new(&logger)?;
            println!("   ✓ Runtime created");

            let engine = runtime.deserialize_cuda_engine(&engine_data)?;
            println!("   ✓ Engine deserialized");

            // Query engine information
            let num_tensors = engine.get_nb_io_tensors()?;
            println!("   ✓ Engine has {} I/O tensors", num_tensors);

            for i in 0..num_tensors {
                let name = engine.get_tensor_name(i)?;
                println!("      - Tensor {}: {}", i, name);
            }

            // Create execution context
            let _context = engine.create_execution_context()?;
            println!("   ✓ Execution context created\n");

            println!("4. Next steps for real inference:");
            println!("   - Allocate CUDA memory for inputs/outputs");
            println!("   - Copy input data to GPU");
            println!("   - Bind tensor addresses with context.set_tensor_address()");
            println!("   - Execute with context.enqueue_v3()");
            println!("   - Copy results back to CPU");
        }
        Err(e) => {
            eprintln!("   ✗ Failed to build engine: {}", e);
            eprintln!("\n   This is expected for an empty network.");
            eprintln!("   In production, add layers before building.");
            return Err(e.into());
        }
    }

    println!("\n✓ Example completed successfully!");

    Ok(())
}
