# Windows GPU Setup Guide for trtx-rs

Complete guide for building and running trtx-rs with NVIDIA TensorRT on Windows with GPU support.

## System Requirements

- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4090)
- **Disk Space**: ~10GB for all dependencies
- **RAM**: 16GB+ recommended

## Prerequisites Installation

### 1. Visual Studio Build Tools 2022

Required for MSVC compiler:

```powershell
# Run as Administrator
winget install Microsoft.VisualStudio.2022.BuildTools
```

During installation, select:
- Desktop development with C++
- MSVC v143 toolset
- Windows 10/11 SDK

### 2. CUDA Toolkit 12.6

Download and install from: https://developer.nvidia.com/cuda-downloads

Or use the automated script:
```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/install-cuda.ps1
```

**Installation Path**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`

### 3. TensorRT 10.7.0.23

Download from NVIDIA Developer: https://developer.nvidia.com/tensorrt

**Installation**: Extract to `D:\TensorRT-10.7.0.23` (or your preferred location)

**Note**: You'll need to create an NVIDIA Developer account to download TensorRT.

### 4. LLVM 19 (for bindgen)

Required for generating Rust bindings from C++ headers:

```powershell
# Download and install LLVM 19
$url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7/LLVM-19.1.7-win64.exe"
$installer = "$env:TEMP\LLVM-19-installer.exe"
Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
Start-Process -FilePath $installer -ArgumentList "/S" -Wait -Verb RunAs
```

**Installation Path**: `C:\Program Files\LLVM`

## Environment Variables

Set the following environment variables (or use the provided scripts):

```powershell
$env:CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:TENSORRT_RTX_DIR = "D:\TensorRT-10.7.0.23"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"

# Add DLL paths for runtime
$env:PATH = "D:\TensorRT-10.7.0.23\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;C:\Program Files\LLVM\bin;$env:PATH"
```

To persist these permanently:
```powershell
[Environment]::SetEnvironmentVariable("CUDA_ROOT", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6", "User")
[Environment]::SetEnvironmentVariable("TENSORRT_RTX_DIR", "D:\TensorRT-10.7.0.23", "User")
[Environment]::SetEnvironmentVariable("LIBCLANG_PATH", "C:\Program Files\LLVM\bin", "User")
```

## Building the Project

### Option 1: Use the provided build script

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/build-simple.ps1
```

### Option 2: Manual build

```powershell
# Set environment variables (see above)
cargo clean
cargo build --release --verbose
```

## Running Tests

### Run all tests with GPU support:

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/test-gpu.ps1
```

### Run specific tests:

```powershell
# Set environment variables first
$env:CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:TENSORRT_RTX_DIR = "D:\TensorRT-10.7.0.23"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:PATH = "D:\TensorRT-10.7.0.23\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;$env:PATH"

# Run tests
cargo test test_device_buffer_allocation
cargo test test_cuda --verbose
```

### Expected Test Results:

✅ **Passing Tests** (8 total):
- `test_device_buffer_allocation` - CUDA memory allocation
- `test_device_buffer_copy` - GPU memory transfers
- `test_synchronize` - CUDA device synchronization
- `test_tensor_input_creation` - Tensor operations
- `test_error_display` - Error handling
- `test_from_ffi` - FFI error conversion
- `test_parse_error_msg` - Error message parsing
- `test_severity_ordering` - Logger severity levels

⚠️ **Ignored Tests** (2 total):
- `test_executor_basic` - Requires ONNX model
- `test_onnx_parser_creation` - Requires TensorRT runtime initialization

## Project Structure

```
trtx-rs/
├── trtx-sys/           # FFI bindings to TensorRT
│   ├── build.rs        # Build script (links TensorRT/CUDA)
│   ├── wrapper.hpp     # C++ header wrapper
│   └── wrapper.cpp     # C++ implementation wrapper
├── trtx/               # Safe Rust API
│   ├── src/
│   │   ├── lib.rs
│   │   ├── logger.rs   # TensorRT logger
│   │   ├── builder.rs  # Engine builder
│   │   ├── runtime.rs  # Runtime for inference
│   │   ├── cuda.rs     # CUDA memory management
│   │   └── onnx_parser.rs
│   └── examples/
│       ├── basic_workflow.rs
│       └── rustnn_executor.rs
├── scripts/            # Build and test automation
│   ├── build-simple.ps1      # Build script
│   ├── test-gpu.ps1          # Test script
│   ├── install-cuda.ps1      # CUDA installation
│   └── download-onnx-model.ps1
└── WINDOWS_GPU_SETUP.md # This file
```

## Troubleshooting

### Issue: `link.exe` fails with "link: extra operand"

**Cause**: Git's `/usr/bin/link.exe` conflicts with MSVC linker

**Solution**: Install Visual Studio Build Tools 2022

### Issue: `LINK : fatal error LNK1181: cannot open input file 'nvinfer.lib'`

**Cause**: TensorRT 10.x uses versioned library names (`nvinfer_10.lib`)

**Solution**: Already fixed in `build.rs` - links to `nvinfer_10` and `nvonnxparser_10`

### Issue: `exit code: 0xc0000135, STATUS_DLL_NOT_FOUND`

**Cause**: Runtime cannot find TensorRT/CUDA DLLs

**Solution**: Add DLL paths to PATH:
```powershell
$env:PATH = "D:\TensorRT-10.7.0.23\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;$env:PATH"
```

### Issue: `error STL1000: Unexpected compiler version`

**Cause**: LLVM version too old for MSVC STL

**Solution**: Upgrade to LLVM 19+

### Issue: `bindgen: Unable to find libclang`

**Cause**: LLVM/Clang not installed or not in PATH

**Solution**: Install LLVM 19 and set `LIBCLANG_PATH`

### Issue: Environment variables not persisting

**Cause**: Variables set in PowerShell session only

**Solution**: Use `[Environment]::SetEnvironmentVariable()` to persist:
```powershell
[Environment]::SetEnvironmentVariable("TENSORRT_RTX_DIR", "D:\TensorRT-10.7.0.23", "User")
```

## Key Implementation Details

### TensorRT 10.x Changes

TensorRT 10 introduced versioned library names:
- `nvinfer.lib` → `nvinfer_10.lib`
- `nvonnxparser.lib` → `nvonnxparser_10.lib`

### Windows vs Linux Paths

The `build.rs` handles platform-specific paths:
- **Windows**: `lib\x64` for CUDA, `/std:c++17` for MSVC
- **Linux**: `lib64` for CUDA, `-std=c++17` for GCC/Clang

### Generated Bindings Differences

Real TensorRT bindings differ from mock bindings:
- **Mock**: `TrtxLoggerSeverity::TRTX_SEVERITY_ERROR`
- **Real**: `TrtxLoggerSeverity_TRTX_SEVERITY_ERROR` (constant)

Error codes:
- Generated as `u32` constants
- Functions return `i32`
- Cast required: `TRTX_ERROR_INVALID_ARGUMENT as i32`

## Performance Notes

- **Build Time**: First build ~5-10 minutes (compiles all dependencies + TensorRT wrapper)
- **Incremental Builds**: ~10-30 seconds
- **Test Runtime**: ~0.2 seconds for all passing tests

## Next Steps

1. **Run Examples**: See example programs in `trtx/examples/`
2. **Build Your Model**: Use the Builder API to create TensorRT engines
3. **Inference**: Use the Runtime API to load engines and run inference
4. **ONNX Support**: Parse ONNX models with the OnnxParser

## Additional Resources

- **TensorRT Documentation**: https://docs.nvidia.com/deeplearning/tensorrt/
- **CUDA Documentation**: https://docs.nvidia.com/cuda/
- **Project Repository**: https://github.com/rustnn/trtx-rs

## License

See LICENSE file in the repository.
