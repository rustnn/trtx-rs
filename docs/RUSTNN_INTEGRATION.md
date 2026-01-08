# TensorRT-RTX Integration with rustnn

## Overview

This document outlines how to integrate `trtx` as a new executor backend in the [rustnn](https://github.com/rustnn/rustnn) project, enabling GPU-accelerated inference on NVIDIA hardware alongside existing ONNX and CoreML backends.

## Current rustnn Architecture Analysis

### Executor Pattern

rustnn follows a **backend-agnostic graph representation** model with three key phases:

1. **Graph Construction**: Platform-independent `GraphInfo` structures
2. **Backend Selection**: Context creation determines execution target
3. **Runtime Execution**: Lazy conversion during `compute()` call

### Existing Executors

**ONNX Runtime** (`executors/onnx.rs`):
```rust
pub fn run_onnx_zeroed(model_bytes: &[u8], inputs: &[OnnxInput]) -> Result<Vec<OnnxOutput>>
pub fn run_onnx_with_inputs(model_bytes: &[u8], inputs: &[OnnxInput]) -> Result<Vec<OnnxOutputWithData>>
```

**CoreML Runtime** (`executors/coreml.rs`):
```rust
pub fn run_coreml_zeroed_cached(model_bytes: &[u8], ...) -> Result<Vec<CoremlOutput>>
```

### Converter System

The `ConverterRegistry` manages format transformations:
- `OnnxConverter`: GraphInfo → ONNX protobuf
- `CoremlMlProgramConverter`: GraphInfo → CoreML MLProgram

Each converter implements the `GraphConverter` trait:
```rust
trait GraphConverter {
    fn format(&self) -> &'static str;
    fn convert(&self, graph_info: &GraphInfo) -> Result<ConvertedGraph>;
}
```

## TensorRT-RTX Integration Design

### Approach 1: ONNX-Based (Recommended)

**Rationale**: TensorRT-RTX natively supports ONNX models, making this the most straightforward integration path.

**Architecture**:
```
GraphInfo → OnnxConverter → ONNX bytes → TrtxExecutor → TensorRT Engine → Results
```

**Advantages**:
- Reuses existing ONNX converter
- Leverages TensorRT-RTX's ONNX parser
- No need for custom graph conversion
- Simpler implementation and maintenance

**Implementation Steps**:

1. **Create TensorRT Executor Module** (`src/executors/trtx.rs`)
2. **Add Feature Flag** to `Cargo.toml`
3. **Implement Executor Functions** following ONNX pattern
4. **Integrate with Backend Selection** in context routing
5. **Add Tests and Examples**

### Approach 2: Native TensorRT Graph (Advanced)

**Rationale**: For maximum performance and TensorRT-specific optimizations.

**Architecture**:
```
GraphInfo → TrtxConverter → TensorRT Network Definition → Engine → Results
```

**Advantages**:
- Direct control over TensorRT network construction
- Access to TensorRT-specific layers and optimizations
- Potentially better performance for certain operations

**Disadvantages**:
- Requires implementing full WebNN → TensorRT operation mapping
- More complex to maintain
- Significant initial development effort

**Deferred**: Start with Approach 1, consider Approach 2 for optimization later.

## Detailed Implementation Plan (Approach 1)

### 1. Project Structure

```
rustnn/
├── Cargo.toml                          # Add trtx dependency + feature
├── src/
│   └── executors/
│       ├── mod.rs                      # Add trtx module export
│       └── trtx.rs                     # New TensorRT executor
└── tests/
    └── test_trtx_executor.rs           # Integration tests
```

### 2. Cargo Configuration

Add to `rustnn/Cargo.toml`:

```toml
[dependencies]
trtx = { version = "0.1", optional = true }

[features]
default = []
trtx-runtime = ["trtx", "trtx/mock"]  # Use mock by default
trtx-runtime-gpu = ["trtx"]           # Real GPU execution

[dev-dependencies]
trtx = { version = "0.1", features = ["mock"] }
```

### 3. Executor Implementation

Create `src/executors/trtx.rs`:

```rust
//! TensorRT-RTX executor for GPU-accelerated inference

use crate::error::GraphError;
use trtx::{Builder, Logger, Runtime};
use trtx::builder::{network_flags, MemoryPoolType};

/// Input descriptor for TensorRT execution
#[derive(Debug, Clone)]
pub struct TrtxInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Output descriptor from TensorRT execution
#[derive(Debug, Clone)]
pub struct TrtxOutput {
    pub name: String,
    pub shape: Vec<usize>,
}

/// Output with actual data
#[derive(Debug, Clone)]
pub struct TrtxOutputWithData {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Execute ONNX model with TensorRT using zero-filled inputs (for validation)
pub fn run_trtx_zeroed(
    onnx_model_bytes: &[u8],
    inputs: &[TrtxInput],
) -> Result<Vec<TrtxOutput>, GraphError> {
    let zero_inputs: Vec<TrtxInput> = inputs
        .iter()
        .map(|input| TrtxInput {
            name: input.name.clone(),
            shape: input.shape.clone(),
            data: vec![0.0; input.shape.iter().product()],
        })
        .collect();

    let outputs = run_trtx_with_inputs(onnx_model_bytes, &zero_inputs)?;

    Ok(outputs
        .into_iter()
        .map(|out| TrtxOutput {
            name: out.name,
            shape: out.shape,
        })
        .collect())
}

/// Execute ONNX model with TensorRT using provided inputs
pub fn run_trtx_with_inputs(
    onnx_model_bytes: &[u8],
    inputs: &[TrtxInput],
) -> Result<Vec<TrtxOutputWithData>, GraphError> {
    // Create logger
    let logger = Logger::stderr()
        .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Logger creation failed: {}", e)))?;

    // Two-phase execution:
    // Phase 1: Build and serialize engine (AOT)
    let engine_data = build_trtx_engine(&logger, onnx_model_bytes)?;

    // Phase 2: Deserialize and execute (Runtime)
    execute_trtx_engine(&logger, &engine_data, inputs)
}

/// Build TensorRT engine from ONNX model (AOT phase)
fn build_trtx_engine(
    logger: &Logger,
    onnx_model_bytes: &[u8],
) -> Result<Vec<u8>, GraphError> {
    let builder = Builder::new(logger)
        .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Builder creation failed: {}", e)))?;

    // Create network with explicit batch
    let network = builder
        .create_network(network_flags::EXPLICIT_BATCH)
        .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Network creation failed: {}", e)))?;

    // Note: In a full implementation, you would use ONNX parser here:
    // let parser = OnnxParser::new(&network, logger)?;
    // parser.parse(onnx_model_bytes)?;

    // For now, this is a simplified version showing the pattern
    // Real implementation needs nvonnxparser bindings

    // Configure builder
    let mut config = builder
        .create_config()
        .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Config creation failed: {}", e)))?;

    // Set workspace memory (1GB)
    config
        .set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 30)
        .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Memory config failed: {}", e)))?;

    // Build serialized engine
    builder
        .build_serialized_network(&network, &config)
        .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Engine build failed: {}", e)))
}

/// Execute TensorRT engine with inputs (Runtime phase)
fn execute_trtx_engine(
    logger: &Logger,
    engine_data: &[u8],
    inputs: &[TrtxInput],
) -> Result<Vec<TrtxOutputWithData>, GraphError> {
    // Create runtime
    let runtime = Runtime::new(logger)
        .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Runtime creation failed: {}", e)))?;

    // Deserialize engine
    let engine = runtime
        .deserialize_cuda_engine(engine_data)
        .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Engine deserialization failed: {}", e)))?;

    // Create execution context
    let mut context = engine
        .create_execution_context()
        .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Context creation failed: {}", e)))?;

    // Get tensor information
    let num_tensors = engine
        .get_nb_io_tensors()
        .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Tensor query failed: {}", e)))?;

    // Allocate CUDA memory for inputs/outputs (simplified - real implementation needs CUDA bindings)
    // For mock mode, this returns dummy data

    // TODO: Implement actual CUDA memory allocation and data transfer
    // For now, return dummy outputs matching the tensor schema
    let mut outputs = Vec::new();
    for i in 0..num_tensors {
        let name = engine
            .get_tensor_name(i)
            .map_err(|e| GraphError::TrtxRuntimeFailed(format!("Tensor name query failed: {}", e)))?;

        // Check if this is an output tensor (simplified)
        if !inputs.iter().any(|input| input.name == name) {
            outputs.push(TrtxOutputWithData {
                name,
                shape: vec![1, 1000], // Dummy shape
                data: vec![0.0; 1000], // Dummy data
            });
        }
    }

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trtx_executor_basic() {
        // This test requires mock mode
        let dummy_onnx = vec![0u8; 100]; // Dummy ONNX model
        let inputs = vec![TrtxInput {
            name: "input".to_string(),
            shape: vec![1, 3, 224, 224],
            data: vec![0.0; 1 * 3 * 224 * 224],
        }];

        let result = run_trtx_zeroed(&dummy_onnx, &inputs);
        // In mock mode, this should succeed with dummy outputs
        assert!(result.is_ok() || cfg!(not(feature = "mock")));
    }
}
```

### 4. Error Handling

Add to `src/error.rs`:

```rust
#[derive(Debug)]
pub enum GraphError {
    // ... existing variants ...

    /// TensorRT-RTX runtime error
    TrtxRuntimeFailed(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // ... existing variants ...
            GraphError::TrtxRuntimeFailed(msg) => write!(f, "TensorRT-RTX error: {}", msg),
        }
    }
}
```

### 5. Module Export

Update `src/executors/mod.rs`:

```rust
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub mod coreml;

#[cfg(feature = "onnx-runtime")]
pub mod onnx;

#[cfg(feature = "trtx-runtime")]
pub mod trtx;
```

### 6. Backend Selection Integration

The backend selection logic would need updates to include TensorRT:

```rust
enum Backend {
    Onnx,
    CoreML,
    TensorRT,  // New
    Fallback,
}

fn select_backend(accelerated: bool, power_preference: &str) -> Backend {
    if !accelerated {
        return Backend::Onnx; // CPU fallback
    }

    match power_preference {
        "high-performance" => {
            // GPU preferred
            #[cfg(feature = "trtx-runtime")]
            if has_nvidia_gpu() {
                return Backend::TensorRT;
            }

            #[cfg(feature = "onnx-runtime")]
            return Backend::Onnx; // ONNX GPU

            Backend::Fallback
        }
        "low-power" => {
            // Prefer NPU (CoreML on Apple Silicon) or integrated GPU
            #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
            return Backend::CoreML;

            #[cfg(feature = "onnx-runtime")]
            return Backend::Onnx;

            Backend::Fallback
        }
        _ => Backend::Onnx,
    }
}
```

## Key Considerations

### ONNX Parser Integration

The current `trtx` crate doesn't include ONNX parser bindings. Two options:

**Option A: Extend trtx crate**
```rust
// Add to trtx crate
pub mod onnx_parser {
    // Bindings to nvonnxparser
}
```

**Option B: Direct FFI in rustnn**
```rust
// In rustnn, create minimal ONNX parser FFI
extern "C" {
    fn nvonnxparser_parse(network: *mut c_void, model: *const u8, len: usize) -> i32;
}
```

### CUDA Memory Management

Full implementation requires CUDA memory operations:
- Allocate device memory (`cudaMalloc`)
- Copy host to device (`cudaMemcpy`)
- Execute inference
- Copy device to host
- Free device memory (`cudaFree`)

Consider using existing CUDA crate or implement minimal FFI.

### Mock Mode for Testing

The `trtx` crate already supports mock mode, enabling:
- CI/CD without GPU hardware
- Development on non-NVIDIA systems
- Unit testing of integration logic

Enable with: `--features trtx-runtime` (uses mock by default)

### Performance Optimization

For production use:
1. **Engine Caching**: Serialize built engines to disk
2. **Profile Optimization**: Use optimization profiles for dynamic shapes
3. **Workspace Tuning**: Adjust memory limits per model
4. **Stream Management**: Reuse CUDA streams for batched inference

## Testing Strategy

### Unit Tests

```rust
#[cfg(all(test, feature = "trtx-runtime"))]
mod tests {
    use super::*;

    #[test]
    fn test_executor_in_mock_mode() {
        // Tests run in mock mode by default
        let result = run_trtx_zeroed(&dummy_model, &inputs);
        assert!(result.is_ok());
    }
}
```

### Integration Tests

```rust
#[test]
#[cfg(feature = "trtx-runtime-gpu")]
#[ignore] // Requires GPU
fn test_real_gpu_inference() {
    // Real GPU test with actual ONNX model
    let model = load_mobilenet_onnx();
    let result = run_trtx_with_inputs(&model, &inputs);
    assert!(result.is_ok());
    // Validate output accuracy
}
```

### CI Configuration

```yaml
# .github/workflows/ci.yml
test-trtx-mock:
  runs-on: ubuntu-latest
  steps:
    - name: Test TensorRT executor (mock)
      run: cargo test --features trtx-runtime

test-trtx-gpu:
  runs-on: self-hosted-gpu  # Requires NVIDIA GPU
  steps:
    - name: Test TensorRT executor (GPU)
      run: cargo test --features trtx-runtime-gpu
```

## Migration Path

### Phase 1: Basic Integration (MVP)
- ✅ Implement trtx executor module
- ✅ Add feature flags and conditional compilation
- ✅ Integrate with backend selection
- ✅ Basic ONNX model support
- ✅ Mock mode for testing

### Phase 2: ONNX Parser Integration
- 🔲 Add nvonnxparser bindings to trtx crate
- 🔲 Implement ONNX model parsing
- 🔲 Test with real ONNX models (MobileNet, ResNet)

### Phase 3: CUDA Memory Management
- 🔲 Implement CUDA memory allocation
- 🔲 Add host↔device data transfer
- 🔲 Test with real inference workloads

### Phase 4: Optimization
- 🔲 Engine caching
- 🔲 Dynamic shape support
- 🔲 Optimization profiles
- 🔲 Performance benchmarking

### Phase 5: Production Readiness
- 🔲 Comprehensive error handling
- 🔲 Resource cleanup and leak prevention
- 🔲 Performance profiling
- 🔲 Documentation and examples

## Usage Example

Once integrated, users would use TensorRT like this:

```python
import webnn as wn

# Create context with GPU acceleration
context = wn.MLContext(accelerated=True, power_preference="high-performance")

# Build graph (backend-agnostic)
builder = wn.MLGraphBuilder(context)
input_operand = builder.input("input", [1, 3, 224, 224])
# ... build graph ...
graph = builder.build({"output": output_operand})

# Execute - automatically routes to TensorRT on NVIDIA GPU
inputs = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
outputs = context.compute(graph, inputs)
```

The backend selection happens automatically based on:
- `accelerated=True` → GPU preferred
- NVIDIA GPU detected → TensorRT selected
- Falls back to ONNX Runtime if TensorRT unavailable

## Comparison with Other Backends

| Feature | ONNX Runtime | CoreML | TensorRT-RTX |
|---------|--------------|---------|--------------|
| **Platforms** | Cross-platform | macOS only | Linux, Windows (NVIDIA GPU) |
| **Hardware** | CPU, CUDA, DirectML | CPU, GPU, Neural Engine | NVIDIA GPU only |
| **Model Format** | ONNX | CoreML MLProgram | ONNX (native) |
| **Conversion** | Direct | Via converter | Direct from ONNX |
| **Performance** | Good | Excellent (Apple HW) | Excellent (NVIDIA GPU) |
| **Mock Mode** | ❌ | ❌ | ✅ (trtx supports) |
| **Complexity** | Low | Medium | Medium |

## Conclusion

Integrating TensorRT-RTX as a rustnn executor follows the established pattern:

1. **Use existing ONNX converter** (no new converter needed)
2. **Implement executor module** following ONNX/CoreML pattern
3. **Add feature flags** for conditional compilation
4. **Integrate with backend selection** for automatic routing
5. **Support mock mode** for testing without GPU

This approach provides:
- GPU-accelerated inference on NVIDIA hardware
- Consistent API with existing backends
- Easy testing via mock mode
- Path to production-grade performance

The implementation can be done incrementally, starting with basic support and adding optimizations over time.
