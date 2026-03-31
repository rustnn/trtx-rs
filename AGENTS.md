# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

trtx-rs provides safe Rust bindings to NVIDIA TensorRT-RTX for high-performance deep learning inference. The project is **experimental** and uses a workspace with two crates:

- **trtx-sys**: Raw FFI bindings (unsafe, internal use only)
- **trtx**: Safe Rust wrapper (user-facing API)

## Build Commands

### Default (Real TensorRT-RTX + cudarc)

By default, trtx builds with real TensorRT-RTX and cudarc for CUDA operations:

```bash
# Set environment
export LD_LIBRARY_PATH=/path/to/tensorrt-rtx/lib

# Build (uses real TensorRT-RTX)
make build                    # Debug build
make build-release            # Release build

# Test
make test                     # Run all tests

# Run example
cargo run --example rustnn_executor
```

### Mock Mode (No GPU Required)

Use mock mode when TensorRT-RTX is not installed (e.g., CI, macOS):

```bash
# Build with mock
make build-mock               # Debug build with mock
cargo build --features mock

# Test with mock
make test-mock                # Verbose output
cargo test --features mock

# Run example
make run-example-mock
cargo run --features mock --example rustnn_executor
```

### Code Quality

```bash
make fmt                     # Format code
make fmt-check              # Check formatting
make clippy                 # Run lints (-D warnings)
make clippy-fix             # Auto-fix warnings
make check-all              # Run fmt-check + clippy + test
```

### Publishing

```bash
make package                # Package both crates
make publish-dry            # Dry-run publish
make publish                # Publish to crates.io
```

## Architecture

### Two-Phase Workflow

**Build Phase (AOT):**
```
Logger → Builder → NetworkDefinition → BuilderConfig → SerializedEngine
```

**Inference Phase (Runtime):**
```
Runtime → Deserialize Engine → ExecutionContext → Bind Tensors → Execute
```

### Core Modules

- `trtx/src/logger.rs`: Logger abstraction with custom handlers
- `trtx/src/builder.rs`: Builder, BuilderConfig, NetworkDefinition
- `trtx/src/runtime.rs`: Runtime, CudaEngine, ExecutionContext
- `trtx/src/onnx_parser.rs`: ONNX model parsing
- `trtx/src/cuda.rs`: CUDA memory management wrappers
- `trtx/src/executor.rs`: High-level executor API (rustnn-compatible)
- `trtx/src/error.rs`: Error types

### Memory Management

- **RAII everywhere**: Use `Drop` trait for automatic cleanup
- **Opaque pointers**: C++ objects exposed as opaque types in Rust
- **CUDA memory**: Wrapped in `DeviceBuffer` with automatic deallocation
- **Error buffers**: Allocated by Rust, passed to C, freed by Rust

### Safety

- Most operations are safe due to RAII
- CUDA operations require `unsafe`:
  - `ExecutionContext::set_tensor_address()` - must point to valid CUDA memory
  - `ExecutionContext::enqueue_v3()` - requires valid CUDA stream
  - CUDA memory operations - explicit device buffer management

### Workspace Dependencies

- Workspace version: 0.2.0
- `trtx-sys` has no dependencies (raw FFI only)
- `trtx` depends on `trtx-sys` and `thiserror`
- Both crates use workspace-level version, edition, license, authors

## Testing Strategy

- Unit tests with mock mode for API validation
- Integration tests for full workflows
- GPU tests on self-hosted runner (Windows + T4 GPU)
- Examples demonstrate real usage patterns

## Environment Variables

- `TENSORRT_RTX_DIR`: TensorRT-RTX installation path (default: `/usr/local/tensorrt-rtx`)
- `CUDA_ROOT`: CUDA installation path (default: `/usr/local/cuda`)
- `LIBCLANG_PATH`: Path to libclang for bindgen (if needed)

## Resources

- Design documentation: `docs/DESIGN.md`
- FFI guide: `docs/FFI_GUIDE.md`
- Integration notes: `docs/RUSTNN_INTEGRATION.md`
- Release process: `docs/HOW_TO_RELEASE.md`
- Windows GPU setup: `docs/WINDOWS_GPU_SETUP.md`
- Pre-commit hooks: `.githooks/` (run `cp .githooks/pre-commit .git/hooks/pre-commit`)
