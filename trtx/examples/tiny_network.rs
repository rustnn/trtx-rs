//! Example: Building and executing a tiny network with the new NetworkDefinition API
//!
//! This example demonstrates:
//! 1. Creating a simple network using the new tensor-based API
//! 2. Building the network into a serialized engine
//! 3. Executing inference with mixed positive/negative input values
//! 4. Verifying ReLU activation behavior (max(0, x))
//!
//! Network architecture:
//! Input [1, 3, 4, 4] -> ReLU -> Output [1, 3, 4, 4]

use trtx::builder::{network_flags, MemoryPoolType};
use trtx::cuda::{synchronize, DeviceBuffer};
use trtx::error::Result;
use trtx::network::Layer; // Import Layer trait for get_output method
use trtx::{Builder, Logger, Runtime};

fn main() -> Result<()> {
    println!("=== Tiny Network Example ===\n");

    // 1. Create logger
    println!("1. Creating logger...");
    let logger = Logger::stderr()?;

    // 2. Build the network
    println!("2. Building network...");
    let engine_data = build_tiny_network(&logger)?;
    println!("   Engine size: {} bytes", engine_data.len());

    // 3. Create runtime and deserialize engine
    println!("\n3. Creating runtime and loading engine...");
    let runtime = Runtime::new(&logger)?;
    let engine = runtime.deserialize_cuda_engine(&engine_data)?;

    // 4. Inspect engine
    println!("4. Engine information:");
    let num_io_tensors = engine.get_nb_io_tensors()?;
    println!("   Number of I/O tensors: {}", num_io_tensors);

    for i in 0..num_io_tensors {
        let name = engine.get_tensor_name(i)?;
        println!("   Tensor {}: {}", i, name);
    }

    // 5. Create execution context
    println!("\n5. Creating execution context...");
    let mut context = engine.create_execution_context()?;

    // 6. Prepare input/output buffers
    println!("6. Preparing buffers...");
    let input_size = 3 * 4 * 4; // [1, 3, 4, 4]
    let output_size = 3 * 4 * 4; // Same as input

    // Create input with mix of positive and negative values
    let input_data: Vec<f32> = (0..input_size)
        .map(|i| {
            // Create pattern: positive, negative, zero, positive, ...
            match i % 4 {
                0 => (i as f32) * 0.5,  // Positive values
                1 => -(i as f32) * 0.3, // Negative values
                2 => 0.0,               // Zero
                _ => (i as f32) * 0.1,  // Small positive values
            }
        })
        .collect();

    println!("   Input shape: [1, 3, 4, 4] ({} elements)", input_size);
    println!("   First 8 input values: {:?}", &input_data[..8]);

    // Allocate device memory
    let mut input_device = DeviceBuffer::new(input_size * std::mem::size_of::<f32>())?;
    let output_device = DeviceBuffer::new(output_size * std::mem::size_of::<f32>())?;

    // Copy input to device (convert f32 slice to bytes)
    let input_bytes = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            input_data.len() * std::mem::size_of::<f32>(),
        )
    };
    input_device.copy_from_host(input_bytes)?;

    // 7. Set tensor addresses
    println!("\n7. Binding tensors...");
    unsafe {
        context.set_tensor_address("input", input_device.as_ptr())?;
        context.set_tensor_address("output", output_device.as_ptr())?;
    }

    // 8. Execute inference
    println!("8. Running inference...");
    let stream = trtx::cuda::get_default_stream();
    unsafe {
        context.enqueue_v3(stream)?;
    }
    synchronize()?;
    println!("   ✓ Inference completed");

    // 9. Copy output back to host
    println!("\n9. Reading results...");
    let mut output_data: Vec<f32> = vec![0.0; output_size];
    let output_bytes = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * std::mem::size_of::<f32>(),
        )
    };
    output_device.copy_to_host(output_bytes)?;

    println!("   Output shape: [1, 3, 4, 4] ({} elements)", output_size);
    println!("   First 8 output values: {:?}", &output_data[..8]);

    // 10. Verify results
    println!("\n10. Verification:");
    println!("   ReLU function: max(0, x)");
    println!("   - Positive inputs should pass through unchanged");
    println!("   - Negative inputs should become 0.0");
    println!("   - Zero inputs should remain 0.0");

    let mut passed = true;
    let mut failures = Vec::new();

    for (i, (&input, &output)) in input_data.iter().zip(output_data.iter()).enumerate() {
        let expected = if input > 0.0 { input } else { 0.0 };
        let diff = (output - expected).abs();

        if diff > 1e-6 {
            passed = false;
            if failures.len() < 5 {
                failures.push((i, input, expected, output));
            }
        }
    }

    if passed {
        println!(
            "\n   ✓ PASS: All {} outputs match expected ReLU behavior!",
            output_size
        );

        // Show some examples
        println!("\n   Sample verification (first 8 elements):");
        for i in 0..8.min(input_size) {
            let input = input_data[i];
            let output = output_data[i];
            let expected = if input > 0.0 { input } else { 0.0 };
            println!(
                "      [{:2}] ReLU({:7.3}) = {:7.3} (expected {:7.3}) ✓",
                i, input, output, expected
            );
        }
    } else {
        println!("\n   ✗ FAIL: {} mismatches found!", failures.len());
        for (i, input, expected, output) in failures {
            println!(
                "      [{:2}] ReLU({:7.3}) = {:7.3}, expected {:7.3}",
                i, input, output, expected
            );
        }
    }

    println!("\n=== Example completed ===");
    Ok(())
}

/// Build a tiny network: Input -> ReLU -> Output
fn build_tiny_network(logger: &Logger) -> Result<Vec<u8>> {
    println!("   Creating builder...");
    let builder = Builder::new(logger)?;

    println!("   Creating network with explicit batch...");
    let mut network = builder.create_network(network_flags::EXPLICIT_BATCH)?;

    println!("   Adding input tensor [1, 3, 4, 4]...");
    // DataType::kFLOAT = 0 in TensorRT
    let input = network.add_input("input", 0, &[1, 3, 4, 4])?;
    println!("   Input tensor name: {:?}", input.name()?);
    println!("   Input tensor dims: {:?}", input.dimensions()?);

    println!("   Adding ReLU activation layer...");
    // ActivationType::kRELU = 0 in TensorRT
    let activation_layer = network.add_activation(&input, 0)?;
    let output = activation_layer.get_output(0)?;

    println!("   Setting output tensor name...");
    let mut output_named = output;
    output_named.set_name("output")?;
    println!("   Output tensor name: {:?}", output_named.name()?);

    println!("   Marking output tensor...");
    network.mark_output(&output_named)?;

    println!("   Network has {} inputs", network.get_nb_inputs());
    println!("   Network has {} outputs", network.get_nb_outputs());

    println!("   Creating builder config...");
    let mut config = builder.create_config()?;

    println!("   Setting memory pool limit (1 GB)...");
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 30)?;

    println!("   Building serialized network...");
    let engine_data = builder.build_serialized_network(&mut network, &mut config)?;

    println!("   ✓ Network built successfully");
    Ok(engine_data)
}
