# trtx-rs

> **⚠️ EXPERIMENTAL - NOT FOR PRODUCTION USE**
>
> This project is in early experimental development. The API is unstable and will change.
> This is NOT production-ready software. Use at your own risk.
>
> Published on crates.io to reserve the crate names.

Safe Rust bindings to [NVIDIA TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX) for high-performance deep learning inference.

## Overview

This project provides ergonomic Rust bindings to TensorRT-RTX, enabling efficient inference of deep learning models on NVIDIA GPUs with minimal overhead.

### Features

- **Safe API**: RAII-based memory management and type-safe abstractions
- **Two-phase workflow**: Separate build (AOT) and inference (runtime) phases
- **Zero-cost abstractions**: Minimal overhead over C++ API
- **Comprehensive error handling**: Proper Rust error types for all operations
- **Flexible logging**: Customizable log handlers for TensorRT messages

## Project Structure

```
trtx-rs/
├── trtx-sys/       # Raw FFI bindings (unsafe)
└── trtx/           # Safe Rust wrapper (use this!)
```

## Prerequisites

### Required

1. **NVIDIA TensorRT-RTX 1.3**: Download and install from [NVIDIA Developer](https://developer.nvidia.com/tensorrt)
2. **CUDA Runtime**: Version compatible with your TensorRT-RTX installation
3. **Clang**: Required for autocxx. On Windows: `winget install LLVM.LLVM`
4. **NVIDIA GPU**: Compatible with TensorRT-RTX requirements

### Environment Setup

Set the installation path if TensorRT-RTX is not in a standard location:

```bash
export TENSORRT_RTX_DIR=/path/to/tensorrt-rtx
export CUDA_ROOT=/usr/local/cuda
```

### Development Without TensorRT-RTX (Mock Mode)

If you're developing on a machine without TensorRT-RTX (e.g., macOS, or for testing), you can use the `mock` feature. This enables the **trtx mock layer** (safe Rust stubs in `trtx` that mirror the real API), not the low-level FFI:

```bash
# Build with mock mode
cargo build --features mock

# Run examples with mock mode
cargo run --features mock --example basic_workflow

# Run tests with mock mode
cargo test --features mock
```

Mock mode provides stub implementations that allow you to:
- Verify the API compiles correctly
- Test your application structure
- Develop without needing an NVIDIA GPU
- Run CI/CD pipelines on any platform

**Note:** Mock mode only validates structure and API usage. For actual inference, you need real TensorRT-RTX.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
trtx = "0.3"
```

## Usage

### Build Phase (Creating an Engine)

```rust
use trtx::{Logger, Builder};
use trtx::builder::{network_flags, MemoryPoolType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create logger
    let logger = Logger::stderr()?;

    // Create builder
    let builder = Builder::new(&logger)?;

    // Create network with explicit batch dimensions
    let network = builder.create_network(network_flags::EXPLICIT_BATCH)?;

    // Configure builder
    let mut config = builder.create_config()?;
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 30)?; // 1GB

    // Build serialized engine
    let engine_data = builder.build_serialized_network(&network, &config)?;

    // Save to disk
    std::fs::write("model.engine", &engine_data)?;

    Ok(())
}
```

### Inference Phase (Running Inference)

```rust
use trtx::{Logger, Runtime};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create logger and runtime
    let logger = Logger::stderr()?;
    let runtime = Runtime::new(&logger)?;

    // Load serialized engine
    let engine_data = fs::read("model.engine")?;
    let engine = runtime.deserialize_cuda_engine(&engine_data)?;

    // Create execution context
    let mut context = engine.create_execution_context()?;

    // Query tensor information
    let num_tensors = engine.get_nb_io_tensors()?;
    for i in 0..num_tensors {
        let name = engine.get_tensor_name(i)?;
        println!("Tensor {}: {}", i, name);
    }

    // Set tensor addresses (requires CUDA memory)
    unsafe {
        context.set_tensor_address("input", input_device_ptr)?;
        context.set_tensor_address("output", output_device_ptr)?;
    }

    // Execute inference
    unsafe {
        context.enqueue_v3(cuda_stream)?;
    }

    Ok(())
}
```

### Custom Logging

```rust
use trtx::{Logger, LogHandler, Severity};

struct MyLogger;

impl LogHandler for MyLogger {
    fn log(&self, severity: Severity, message: &str) {
        match severity {
            Severity::Error | Severity::InternalError => {
                eprintln!("ERROR: {}", message);
            }
            Severity::Warning => {
                println!("WARN: {}", message);
            }
            _ => {
                println!("INFO: {}", message);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let logger = Logger::new(MyLogger)?;
    // Use logger...
    Ok(())
}
```

## API Overview

### Core Types

- **`Logger`**: Captures TensorRT messages with custom handlers
- **`Builder`**: Creates optimized inference engines
- **`NetworkDefinition`**: Defines the computational graph
- **`BuilderConfig`**: Configures optimization parameters
- **`Runtime`**: Deserializes engines for inference
- **`CudaEngine`**: Optimized inference engine
- **`ExecutionContext`**: Manages inference execution

### Error Handling

All fallible operations return `Result<T, Error>`:

```rust
use trtx::Error;

match builder.create_network(0) {
    Ok(network) => {
        // Use network
    }
    Err(Error::InvalidArgument(msg)) => {
        eprintln!("Invalid argument: {}", msg);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Safety

### Safe Operations

Most operations are safe and use RAII for resource management:
- Creating loggers, builders, runtimes
- Building and serializing engines
- Deserializing engines
- Creating execution contexts

### Unsafe Operations

CUDA-related operations require `unsafe`:
- **`set_tensor_address`**: Must point to valid CUDA device memory
- **`enqueue_v3`**: Requires valid CUDA stream and properly bound tensors

## Building from Source

```bash
# Clone the repository
git clone https://github.com/rustnn/trtx-rs.git
cd trtx-rs

# Option 1: Build with TensorRT-RTX (requires NVIDIA GPU)
export TENSORRT_RTX_DIR=/path/to/tensorrt-rtx
cargo build --release
cargo test

# Option 2: Build in mock mode (no GPU required)
cargo build --features mock --release
cargo test --features mock
cargo run --features mock --example basic_workflow
```

## Examples

See the `trtx/examples/` directory for complete examples:

- `basic_workflow.rs`: Build and serialize an engine (optionally from ONNX), then run inference
- `tiny_network.rs`: Build a small ReLU-based network from scratch using the Network API (no ONNX)
- `rustnn_executor.rs`: rustnn-compatible executor integration

## Architecture

### trtx-sys (FFI Layer)

- **autocxx**-generated bindings for the TensorRT-RTX C++ API
- Slim C++ logger bridge for virtual method handling (e.g., log callbacks)
- Optional mock FFI (when `mock` feature is enabled) so the crate can build without TensorRT installed
- No safety guarantees; internal use only

### trtx (Safe Wrapper)

- **Mock layer**: When the `mock` feature is enabled, the trtx crate uses a Rust mock layer (`trtx/src/mock/`) that mirrors the real API—this is the “mock mode” you use for development without GPU. Real implementation lives in `trtx/src/real/`.
- RAII-based resource management
- Type-safe API
- Lifetime tracking
- Comprehensive error handling
- User-facing API

## Troubleshooting

### Build Errors

**Cannot find TensorRT headers:**
```bash
export TENSORRT_RTX_DIR=/path/to/tensorrt-rtx
```

**Linking errors:**
```bash
export LD_LIBRARY_PATH=$TENSORRT_RTX_DIR/lib:$LD_LIBRARY_PATH
```

### Runtime Errors

**CUDA not initialized:**
Ensure CUDA runtime is properly initialized before creating engines or contexts.

**Invalid tensor addresses:**
Verify that all tensor addresses point to valid CUDA device memory with correct sizes.

## Development

### Pre-commit Hooks

To ensure code quality, set up the pre-commit hook:

```bash
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The hook will automatically run `cargo fmt` and `cargo clippy` before each commit.

### Manual Checks

You can also run checks manually using the Makefile:

```bash
make check-all  # Run fmt, clippy, and tests
make fmt        # Format code
make clippy     # Run lints
make test       # Run tests
```

### GPU Testing

The project includes CI workflows for testing with real NVIDIA GPUs:

- **Mock mode CI**: Runs on every push (ubuntu, macos) - tests API without GPU
- **GPU tests**: Runs on self-hosted Windows runner with T4 GPU - tests real TensorRT-RTX

To set up a GPU runner for real hardware testing, see [GPU Runner Setup Guide](.github/GPU_RUNNER_SETUP.md).

The GPU tests workflow:
- Builds without mock mode (uses real TensorRT-RTX)
- Verifies CUDA and GPU availability
- Runs tests and examples with actual GPU acceleration
- Can be triggered manually or runs automatically on code changes

## Contributing

Contributions are welcome! Please see [docs/DESIGN.md](docs/DESIGN.md) for architecture details.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA for TensorRT-RTX
- The Rust community for excellent FFI tools

## Status

This project is in early development. APIs may change before 1.0 release.

### Implemented

- ✅ Core FFI layer (autocxx); trtx **mock layer** for development without TensorRT (no GPU)
- ✅ Logger interface with custom handlers
- ✅ Builder API for engine creation
- ✅ Runtime and engine deserialization
- ✅ Execution context
- ✅ Error handling with detailed messages
- ✅ **Network API**: TensorRT-RTX `INetworkDefinition` supported—build networks in Rust without ONNX
- ✅ **ONNX parser bindings** (nvonnxparser integration)
- ✅ **CUDA**: cudarc integration for memory management and device sync
- ✅ **rustnn-compatible executor API** (ready for integration)
- ✅ RAII-based resource management

### Planned

- ⬜ Dynamic shape support
- ⬜ Optimization profiles
- ⬜ Weight refitting
- ⬜ INT8 quantization support
- ⬜ Comprehensive examples with real models
- ⬜ Performance benchmarking
- ⬜ Documentation improvements

## Resources

- [TensorRT-RTX Documentation](https://docs.nvidia.com/deeplearning/tensorrt-rtx/)
- [TensorRT-RTX GitHub](https://github.com/NVIDIA/TensorRT-RTX)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
