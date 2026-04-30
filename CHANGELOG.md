# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-04-30

### Added
- trtexec-rs as a Rust port of trtexec (supports also WebNN graphs)
- Support for IProfiler interface
- More methods on ExecutionContext, CudaEngine and some Layers
- Add support for setting NCCL communicator on networks

### Changed
- Removed `get_` prefix from getters (old names deprecated)

### Fixed
- Fixed concurrent dynamic loading (only load libraries once)
- Make sure OnnxParser can't be dropped before build
- Add lifetimes of parent object to BuilderConfig


## [0.4.0] - 2026-04-09

### Added
- Support for TensorRT RTX 1.4 (see https://docs.nvidia.com/deeplearning/tensorrt-rtx/1.4/index.html#what-s-new-in-tensorrt-rtx-1-4)
- Refitter API
- IProgressMonitor, IErrorRecorder, IDebug, IGpuAllocator, IDebugListener exposed
- More layers and layer methods exposed

### Changed
- trtx now enforces with lifetimes correct usage of API
- autocxx is now used for almost everything, even mock
- mock now uses the same API as real mode, builder API now has limited mock coverage

### Fixed
- fixed one SEGFAULT in `add_comulative` (now prevented by lifetimes in API)

## [0.3.1] - 2026-02-26

### Added

- Added missing layer and tensor capabilities needed by RustNN: layer naming/introspection (`set_layer_name`, `get_layer_name`, `get_layer_type`, `get_nb_layers`), tensor format control (`set_allowed_formats`), engine tensor dtype lookup, convolution/deconvolution input setters, and expanded deconvolution controls (stride/padding/dilation/groups). Mock mode now mirrors the new APIs.
- Extended FFI bindings with TensorRT `TensorFormat` and related enums so the safe API can expose format/weight type choices.

### Changed

- TensorRT libraries now auto-load on first use (builder, ONNX parser, runtime) and guard against double-loading when using `dlopen` mode.
- Build script normalizes bundled TensorRT headers (size_t fixes, doxygen cleanup) to improve autocxx generation and rustdoc quality; future header drops are transformed automatically.
- Dependency versions bumped to current releases; crate-level docs refreshed to align with README.

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

[0.4.0]: https://github.com/rustnn/trtx-rs/releases/tag/v0.4.0
[0.3.0]: https://github.com/rustnn/trtx-rs/releases/tag/v0.3.0
[0.3.1]: https://github.com/rustnn/trtx-rs/releases/tag/v0.3.1
