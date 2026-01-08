# TensorRT-RTX Rust Bindings Design

## Overview

This project provides Rust bindings for NVIDIA TensorRT-RTX, split into two crates:
- **trtx-sys**: Raw FFI bindings generated via bindgen
- **trtx**: Safe, ergonomic Rust wrapper

## API Analysis

### Core Components

Based on the C++ API, the main interfaces are:

1. **ILogger**: Message capture at different severity levels
2. **IBuilder**: Creates optimized inference engines
3. **INetworkDefinition**: Represents the computational graph
4. **IBuilderConfig**: Specifies optimization parameters
5. **ICudaEngine**: Holds the optimized model (serializable)
6. **IExecutionContext**: Manages inference state and activations
7. **IRuntime**: Deserializes engines for inference

### Typical Workflow

**Build Phase (AOT):**
```
Logger → Builder → NetworkDefinition → BuilderConfig → SerializedEngine
```

**Inference Phase (Runtime/JIT):**
```
Runtime → Deserialize Engine → ExecutionContext → Bind Tensors → Execute
```

## Project Structure

```
trtx-rs/
├── Cargo.toml                    # Workspace manifest
├── trtx-sys/                     # Raw FFI layer
│   ├── Cargo.toml
│   ├── build.rs                  # Bindgen configuration
│   ├── src/
│   │   └── lib.rs                # Raw bindings
│   └── wrapper.hpp               # C++ wrapper headers (if needed)
├── trtx/                         # Safe Rust wrapper
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                # Public API
│   │   ├── logger.rs             # Logger abstraction
│   │   ├── builder.rs            # Builder API
│   │   ├── network.rs            # Network definition
│   │   ├── config.rs             # Builder configuration
│   │   ├── engine.rs             # Engine management
│   │   ├── runtime.rs            # Runtime API
│   │   ├── context.rs            # Execution context
│   │   ├── tensor.rs             # Tensor bindings
│   │   ├── buffer.rs             # Memory management
│   │   ├── error.rs              # Error types
│   │   └── cuda.rs               # CUDA utilities
│   ├── examples/
│   │   ├── build_engine.rs
│   │   └── run_inference.rs
│   └── tests/
│       └── integration_tests.rs
├── README.md
└── DESIGN.md                     # This file
```

## Design Decisions

### 1. FFI Layer (trtx-sys)

**Approach:**
- Use bindgen to generate raw bindings
- Minimal manual intervention
- Direct mapping to C++ API
- No safety guarantees

**Challenges:**
- C++ classes → Opaque types in Rust
- Need C wrapper functions for:
  - Constructor/destructor calls
  - Virtual method dispatch
  - Exception handling

**Build Configuration:**
```rust
// build.rs
bindgen::Builder::default()
    .header("wrapper.hpp")
    .allowlist_type("nvinfer1::.*")
    .allowlist_function("nvinfer1::.*")
    .opaque_type("std::.*")
    .generate_cstr(true)
    .parse_callbacks(Box::new(bindgen::CargoCallbacks))
```

### 2. Safe Wrapper (trtx)

**Design Principles:**
1. **RAII**: Automatic resource cleanup via Drop
2. **Type Safety**: Leverage Rust's type system
3. **Error Handling**: Result<T, Error> for all fallible operations
4. **Builder Pattern**: Ergonomic configuration
5. **Lifetime Safety**: Enforce object dependencies at compile time

**Key Types:**

```rust
// Logger
pub struct Logger { /* ... */ }
impl ILogger for Logger { /* ... */ }

// Builder
pub struct Builder<'a> {
    logger: &'a Logger,
    // ...
}

// Engine (owns serialized data)
pub struct Engine {
    data: Vec<u8>,
    // ...
}

// Runtime
pub struct Runtime<'a> {
    logger: &'a Logger,
    // ...
}

// ExecutionContext (borrows Engine)
pub struct ExecutionContext<'a> {
    engine: &'a Engine,
    // ...
}
```

**Memory Management:**
- Use `Box<T>` for heap-allocated objects
- CUDA memory via wrapper types with Drop
- Explicit lifetime management for borrowed resources

**Error Handling:**

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Builder error: {0}")]
    Builder(String),

    #[error("Engine error: {0}")]
    Engine(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

### 3. API Surface

**Minimal Initial API:**

```rust
// Build phase
let logger = Logger::new(Severity::Info);
let builder = Builder::new(&logger)?;
let network = builder.create_network()?;

// Add layers to network...
network.add_input("input", Dims::new(&[1, 3, 224, 224]), DataType::Float)?;
// ...

let config = BuilderConfig::new()?;
config.set_max_workspace_size(1 << 30)?;

let engine_data = builder.build_serialized_network(&network, &config)?;
std::fs::write("model.engine", &engine_data)?;

// Inference phase
let runtime = Runtime::new(&logger)?;
let engine = runtime.deserialize_engine(&engine_data)?;
let context = engine.create_execution_context()?;

// Bind tensors
context.set_tensor_address("input", input_buffer.as_ptr())?;
context.set_tensor_address("output", output_buffer.as_mut_ptr())?;

// Execute
context.enqueue_v3(&cuda_stream)?;
cuda_stream.synchronize()?;
```

### 4. CUDA Integration

**Options:**
1. **Minimal**: Assume user manages CUDA, accept raw pointers
2. **Integrated**: Provide safe wrappers for common CUDA operations
3. **Depend on existing crate**: Use `cudarc` or similar

**Recommendation**: Start with option 1, expand to 2 as needed.

### 5. C++ Exception Handling

**Problem**: C++ API throws exceptions, Rust FFI is unsafe

**Solution**: Create C wrapper functions that catch exceptions and return error codes:

```cpp
// wrapper.hpp
extern "C" {
    int32_t trtx_create_builder(
        void* logger,
        void** out_builder,
        char* error_msg,
        size_t error_msg_len
    );
}
```

```rust
// trtx-sys
pub unsafe fn create_builder(
    logger: *mut c_void
) -> Result<*mut c_void, String> {
    let mut builder: *mut c_void = std::ptr::null_mut();
    let mut error_msg = [0i8; 1024];

    let result = trtx_create_builder(
        logger,
        &mut builder,
        error_msg.as_mut_ptr(),
        error_msg.len(),
    );

    if result == 0 {
        Ok(builder)
    } else {
        Err(/* convert error_msg to String */)
    }
}
```

## Implementation Plan

### Phase 1: Foundation
1. Create workspace structure
2. Set up trtx-sys with bindgen
3. Create C++ wrapper for critical functions
4. Implement basic Logger binding
5. Test build system

### Phase 2: Core API
1. Implement Builder and BuilderConfig
2. Implement NetworkDefinition basics
3. Implement Engine serialization/deserialization
4. Implement Runtime
5. Implement ExecutionContext

### Phase 3: Inference
1. Implement tensor binding API
2. Add CUDA stream support
3. Create buffer management utilities
4. Write basic inference example

### Phase 4: Advanced Features
1. Dynamic shapes support
2. Optimization profiles
3. Weight refitting
4. CUDA graphs
5. Multi-context inference

## Dependencies

### trtx-sys
```toml
[build-dependencies]
bindgen = "0.70"
cc = "1.0"

[dependencies]
# None - just raw FFI
```

### trtx
```toml
[dependencies]
trtx-sys = { path = "../trtx-sys" }
thiserror = "1.0"

[dev-dependencies]
# For examples
```

## Testing Strategy

1. **Unit tests**: Test individual components with mock engines
2. **Integration tests**: Test full build → inference pipeline
3. **Examples**: Demonstrate real usage patterns
4. **Documentation tests**: Ensure API examples compile

## Open Questions

1. **TensorRT-RTX installation**: How should users install the library?
   - System-wide installation?
   - Bundled with crate?
   - Document as prerequisite?

2. **Platform support**: Linux-only? Windows? macOS (if RTX supports)?

3. **CUDA version requirements**: What CUDA versions are supported?

4. **License compatibility**: TensorRT-RTX license vs Rust crate license

5. **API coverage**: Start with minimal API or comprehensive coverage?
   - Recommendation: Minimal first, expand based on usage

## Next Steps

1. Review this design with stakeholders
2. Set up project structure
3. Verify TensorRT-RTX installation and headers
4. Begin Phase 1 implementation
