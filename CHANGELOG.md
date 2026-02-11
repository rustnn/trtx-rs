# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-02-11

### Added

- **Dependency-Free Build**: Included TensorRT-RTX headers and provided internal typedefs for essential CUDA types. This allows the crate to be built out-of-the-box without requiring a pre-installed CUDA Toolkit or TensorRT-RTX SDK.
- **Lazy DLL Loading**: Implemented delay-loading for TensorRT-RTX shared libraries. This allows applications to link against trtx-rs and start up even on systems where TensorRT-RTX is not supported while at the same time utilize TensorRT-RTX on system with supported GPUs.
- **Network Definition API**: The TensorRT-RTX `INetworkDefinition` interface is now supported, allowing networks to be built programmatically in Rust without ONNX.
- **Tiny network example**: New example (`examples/tiny_network.rs`) demonstrating creation of a ReLU-based network from scratch.

### Changed

- **Overhauled FFI Layer**: Migrated from manual bindgen headers to an autocxx-based architecture. This automates safe C++ interop for the core TensorRT-RTX API and significantly reduces unsafe boilerplate, while utilizing a specialized C++ shim only for complex virtual method handling (e.g., Loggers).
- **CUDA bindings**: Integrated the cudarc crate for CUDA memory management and device synchronization.
- **Improved mock mode**: Overhauled mock implementations to provide stubbed types (e.g., Dims64, nvinfer1::DataType) allowing full compilation on macOS/CI without TensorRT or NVIDIA GPUs.

### Fixed

- **Environment discovery**: Improved `build.rs` to more reliably find `TENSORRT_RTX_DIR` and `CUDA_ROOT` across Windows and Linux.

---

### Highlights for 0.3.0

- **Network Definition Interface**: Exposed the core TensorRT network building API, enabling the construction of custom graph topologies (inputs, layers, and outputs) directly from Rust.
- **CUDA Integration**: Migrated CUDA memory and stream management to cudarc to align with modern Rust GPU tooling.
- **Improved Portability**: Expanded the mock mode to include stubbed TensorRT types, allowing the library to be compiled non-NVIDIA platforms (e.g., macOS).

[0.3.0]: https://github.com/rustnn/trtx-rs/releases/tag/v0.3.0
