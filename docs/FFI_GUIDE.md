# FFI Bindings Guide

This document explains how the FFI bindings for TensorRT-RTX are built and how to update them when the C++ API changes.

## Architecture Overview

The FFI layer uses a **three-layer architecture**:

```
┌─────────────────────────┐
│  Rust Safe API (trtx)   │  <- High-level safe Rust
├─────────────────────────┤
│  Raw FFI (trtx-sys)     │  <- Auto-generated bindings
├─────────────────────────┤
│  C Wrapper Layer        │  <- Hand-written C interface
│  (wrapper.hpp/cpp)      │
├─────────────────────────┤
│  TensorRT-RTX C++ API   │  <- NVIDIA's C++ library
└─────────────────────────┘
```

### Why Three Layers?

1. **C Wrapper Layer** (`wrapper.hpp` + `wrapper.cpp`)
   - TensorRT-RTX is a C++ API that cannot be directly bound by Rust
   - Provides a C-compatible interface with `extern "C"` linkage
   - Handles C++ exceptions and converts them to error codes
   - Uses opaque pointer types to hide C++ objects

2. **Bindgen Layer** (automatic)
   - Reads `wrapper.hpp` header file
   - Automatically generates Rust FFI declarations
   - Outputs to `target/debug/build/trtx-sys-*/out/bindings.rs`

3. **Safe Rust API** (in `trtx` crate)
   - Wraps unsafe FFI calls in safe Rust abstractions
   - Provides RAII types, Result-based error handling
   - Adds Rust idioms and type safety

## Build Process

The build happens in `build.rs`:

### 1. Compile C++ Wrapper

```rust
// Build C++ wrapper
let mut build = cc::Build::new();
build.cpp(true)
    .file("wrapper.cpp")
    .include(&include_dir)
    .flag("-std=c++17");

build.compile("trtx_wrapper");
```

This compiles `wrapper.cpp` into a static library that links against TensorRT-RTX.

### 2. Generate Rust Bindings

```rust
let bindings = bindgen::Builder::default()
    .header("wrapper.hpp")                          // Input C header
    .clang_arg(format!("-I{}", include_dir))       // Include TensorRT headers
    .allowlist_function("trtx_.*")                 // Only expose trtx_* functions
    .allowlist_type("TrtxLogger.*")                // Only expose Trtx* types
    .allowlist_var("TRTX_.*")                      // Only expose TRTX_* constants
    .derive_debug(true)                            // Add Debug trait
    .derive_default(true)                          // Add Default trait
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .generate()
    .expect("Unable to generate bindings");

bindings
    .write_to_file(out_path.join("bindings.rs"))
    .expect("Couldn't write bindings!");
```

### 3. Link Libraries

```rust
println!("cargo:rustc-link-search=native={}", lib_dir);
println!("cargo:rustc-link-lib=dylib=nvinfer_10");
println!("cargo:rustc-link-lib=dylib=nvonnxparser_10");
println!("cargo:rustc-link-lib=dylib=cudart");
```

## How to Update Bindings When C++ API Changes

### Scenario 1: Adding New TensorRT-RTX Functions

**Example**: You want to expose `ICudaEngine::getTensorShape()`

#### Step 1: Add C wrapper function in `wrapper.hpp`

```c
int32_t trtx_cuda_engine_get_tensor_shape(
    TrtxCudaEngine* engine,
    const char* tensor_name,
    int32_t* out_dims,
    int32_t* out_nb_dims,
    char* error_msg,
    size_t error_msg_len
);
```

#### Step 2: Implement in `wrapper.cpp`

```cpp
int32_t trtx_cuda_engine_get_tensor_shape(
    TrtxCudaEngine* engine_ptr,
    const char* tensor_name,
    int32_t* out_dims,
    int32_t* out_nb_dims,
    char* error_msg,
    size_t error_msg_len
) {
    try {
        auto* engine = reinterpret_cast<nvinfer1::ICudaEngine*>(engine_ptr);

        nvinfer1::Dims dims = engine->getTensorShape(tensor_name);
        *out_nb_dims = dims.nbDims;

        for (int i = 0; i < dims.nbDims; i++) {
            out_dims[i] = dims.d[i];
        }

        return TRTX_SUCCESS;
    } catch (const std::exception& e) {
        copy_error_msg(error_msg, error_msg_len, e.what());
        return TRTX_ERROR_RUNTIME_ERROR;
    }
}
```

#### Step 3: Rebuild to regenerate bindings

```bash
cargo clean -p trtx-sys
cargo build
```

Bindgen automatically picks up the new function from `wrapper.hpp` and generates the Rust FFI declaration.

#### Step 4: Add safe wrapper in `trtx/src/engine.rs`

```rust
impl CudaEngine {
    pub fn tensor_shape(&self, name: &str) -> Result<Vec<i32>> {
        let c_name = CString::new(name)?;
        let mut dims = [0i32; 8];  // Max 8 dimensions
        let mut nb_dims = 0i32;

        unsafe {
            check_error(trtx_cuda_engine_get_tensor_shape(
                self.ptr,
                c_name.as_ptr(),
                dims.as_mut_ptr(),
                &mut nb_dims,
                error_msg.as_mut_ptr(),
                ERROR_MSG_LEN,
            ))?;
        }

        Ok(dims[..nb_dims as usize].to_vec())
    }
}
```

### Scenario 2: Updating Existing Function Signatures

**Example**: TensorRT changes `setMemoryPoolLimit()` signature

#### Step 1: Update `wrapper.hpp` signature

```c
// Old
int32_t trtx_builder_config_set_memory_pool_limit(
    TrtxBuilderConfig* config,
    int32_t pool_type,
    size_t pool_size,
    char* error_msg,
    size_t error_msg_len
);

// New - added pool_flags parameter
int32_t trtx_builder_config_set_memory_pool_limit(
    TrtxBuilderConfig* config,
    int32_t pool_type,
    size_t pool_size,
    uint32_t pool_flags,
    char* error_msg,
    size_t error_msg_len
);
```

#### Step 2: Update implementation in `wrapper.cpp`

```cpp
int32_t trtx_builder_config_set_memory_pool_limit(
    TrtxBuilderConfig* config_ptr,
    int32_t pool_type,
    size_t pool_size,
    uint32_t pool_flags,
    char* error_msg,
    size_t error_msg_len
) {
    try {
        auto* config = reinterpret_cast<nvinfer1::IBuilderConfig*>(config_ptr);
        config->setMemoryPoolLimit(
            static_cast<nvinfer1::MemoryPoolType>(pool_type),
            pool_size,
            pool_flags  // New parameter
        );
        return TRTX_SUCCESS;
    } catch (const std::exception& e) {
        copy_error_msg(error_msg, error_msg_len, e.what());
        return TRTX_ERROR_RUNTIME_ERROR;
    }
}
```

#### Step 3: Rebuild

```bash
cargo clean -p trtx-sys
cargo build
```

This will fail if the high-level `trtx` crate still uses the old signature.

#### Step 4: Update all call sites in `trtx` crate

Find and update all uses:

```bash
# Find all uses
rg "trtx_builder_config_set_memory_pool_limit" trtx/src/

# Update each call site to pass the new parameter
```

### Scenario 3: Adding Mock Support for New Functions

When adding new functions, also update the mock bindings for development without TensorRT.

#### Update `build.rs` mock bindings

In the `generate_mock_bindings()` function, add:

```rust
fn generate_mock_bindings(out_path: &Path) {
    let mock_bindings = r#"
    // ... existing mock code ...

    // Add new function declaration
    extern "C" {
        pub fn trtx_cuda_engine_get_tensor_shape(
            engine: *mut TrtxCudaEngine,
            tensor_name: *const ::std::os::raw::c_char,
            out_dims: *mut i32,
            out_nb_dims: *mut i32,
            error_msg: *mut ::std::os::raw::c_char,
            error_msg_len: usize,
        ) -> i32;
    }
    "#;

    std::fs::write(out_path.join("bindings.rs"), mock_bindings)
        .expect("Couldn't write mock bindings!");
}
```

#### Update `mock.c` implementation

```c
int32_t trtx_cuda_engine_get_tensor_shape(
    TrtxCudaEngine* engine,
    const char* tensor_name,
    int32_t* out_dims,
    int32_t* out_nb_dims,
    char* error_msg,
    size_t error_msg_len
) {
    // Mock implementation: return dummy shape
    *out_nb_dims = 3;
    out_dims[0] = 1;
    out_dims[1] = 224;
    out_dims[2] = 224;
    return TRTX_SUCCESS;
}
```

## Testing Changes

### 1. Test with Mock Mode

```bash
cargo test --features mock
```

### 2. Test with Real TensorRT-RTX

```bash
TENSORRT_RTX_DIR=/usr/local/tensorrt-rtx cargo test
```

### 3. Check Generated Bindings

```bash
# View generated bindings
cat target/debug/build/trtx-sys-*/out/bindings.rs | less

# Search for your new function
rg "trtx_cuda_engine_get_tensor_shape" target/debug/build/trtx-sys-*/out/bindings.rs
```

## Common Issues

### Issue 1: Bindgen Can't Find Headers

**Error**: `fatal error: 'NvInfer.h' file not found`

**Solution**: Set `TENSORRT_RTX_DIR` or update include path in `build.rs`:

```bash
export TENSORRT_RTX_DIR=/path/to/tensorrt-rtx
cargo build
```

### Issue 2: Linker Can't Find Libraries

**Error**: `ld: library not found for -lnvinfer_10`

**Solution**: Check library search path and library names:

```bash
ls $TENSORRT_RTX_DIR/lib/
# Update build.rs if library names have changed
```

### Issue 3: C++ Exceptions Crash Rust

**Problem**: C++ exception crosses FFI boundary

**Solution**: Always wrap C++ code in try-catch in `wrapper.cpp`:

```cpp
try {
    // C++ code that might throw
} catch (const std::exception& e) {
    copy_error_msg(error_msg, error_msg_len, e.what());
    return TRTX_ERROR_RUNTIME_ERROR;
}
```

### Issue 4: Function Not Exposed to Rust

**Problem**: New function in wrapper.hpp but not appearing in Rust

**Solution**: Check bindgen allowlist patterns in `build.rs`:

```rust
.allowlist_function("trtx_.*")  // Must match your function name
```

## Best Practices

### Naming Conventions

- **C functions**: `trtx_<class>_<method>` (e.g., `trtx_cuda_engine_get_tensor_shape`)
- **Types**: `Trtx<ClassName>` (e.g., `TrtxCudaEngine`)
- **Constants**: `TRTX_<NAME>` (e.g., `TRTX_SUCCESS`)

### Error Handling Pattern

All FFI functions should follow this pattern:

```c
int32_t trtx_function_name(
    // ... input parameters ...
    char* error_msg,        // Always second-to-last
    size_t error_msg_len    // Always last
);
```

Return values:
- `TRTX_SUCCESS` (0) on success
- Error code on failure (error message written to `error_msg`)

### Memory Management

- **Rust owns**: Rust-allocated memory passed to C (must free in Rust)
- **C owns**: C-allocated objects (use destroy functions to free)
- **Return buffers**: C allocates, Rust calls `trtx_free_buffer()` to free

Example:
```rust
// C allocates buffer
let mut data: *mut c_void = std::ptr::null_mut();
let mut size: usize = 0;
trtx_builder_build_serialized_network(..., &mut data, &mut size, ...);

// Rust must free when done
trtx_free_buffer(data);
```

### Type Safety

Use opaque pointer types in C:
```c
typedef struct TrtxCudaEngine TrtxCudaEngine;  // Opaque
```

Cast to real type only in C++:
```cpp
auto* engine = reinterpret_cast<nvinfer1::ICudaEngine*>(engine_ptr);
```

## Development Workflow

### Quick Reference

```bash
# 1. Modify wrapper.hpp and wrapper.cpp
vim wrapper.hpp wrapper.cpp

# 2. Rebuild bindings
cargo clean -p trtx-sys && cargo build

# 3. Check generated bindings
cat target/debug/build/trtx-sys-*/out/bindings.rs | rg "your_function"

# 4. Update safe Rust wrapper in trtx crate
vim ../trtx/src/your_module.rs

# 5. Test
cargo test

# 6. Test with mock
cargo test --features mock
```

## References

- [Bindgen User Guide](https://rust-lang.github.io/rust-bindgen/)
- [The Rustonomicon - FFI](https://doc.rust-lang.org/nomicon/ffi.html)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

## Questions?

If you encounter issues not covered here, check:

1. Bindgen output for errors: `cargo build -vv`
2. Generated bindings: `cat target/debug/build/trtx-sys-*/out/bindings.rs`
3. Linker output: `cargo build -vv 2>&1 | grep "ld:"`
