/**
 * Logger Bridge for TensorRT-RTX Rust Bindings
 *
 * This file provides C wrapper functions for TensorRT-RTX C++ API.
 * While we use autocxx for most C++ bindings, some wrappers are still
 * necessary.
 *
 * ## Architecture
 *
 * ```
 * Rust (trtx) → Raw FFI (trtx-sys) → logger_bridge.cpp → TensorRT C++ + autocxx
 * ```
 *
 * ## Why These Wrappers Exist
 *
 * ### NECESSARY WRAPPERS (Cannot be removed):
 *
 * 1. **Logger Bridge (lines 29-52)**:
 *    - Rust cannot implement C++ virtual classes
 *    - RustLoggerImpl forwards virtual method calls to Rust callbacks
 *    - REQUIRED: No alternative
 *
 * 2. **Factory Functions (lines 55-91)**:
 *    - createInferBuilder/Runtime take `ILogger&` references
 *    - autocxx struggles with C++ reference parameters
 *    - REQUIRED: Simplest solution for reference params
 *
 * 3. **CUDA Wrappers (lines 658-677)**:
 *    - Bridge between std::ffi::c_void and autocxx::c_void
 *    - Type compatibility issue between Rust and autocxx types
 *    - KEEP FOR NOW: Could be removed with codebase-wide type migration
 *
 * ### POTENTIALLY REDUNDANT WRAPPERS:
 *
 * 4. **TensorRT Method Wrappers (lines 94-657)**:
 *    - Builder, Network, Tensor, Engine, Context methods
 *    - autocxx CAN generate these with
 * `generate!("nvinfer1::INetworkDefinition")`
 *    - POTENTIALLY REMOVABLE: ~75% code reduction if refactored
 *    - STATUS: Kept for now due to stability, could be migrated to direct
 * autocxx calls
 *
 * ## Why Not Full autocxx?
 *
 * We TRIED to use autocxx for everything but encountered:
 * - Type mismatches (autocxx::c_void vs std::ffi::c_void)
 * - Reference parameter handling issues
 * - Virtual method/callback complications
 *
 * ## See Also
 * - docs/LOGGER_BRIDGE_ANALYSIS.md - Detailed analysis of each function
 * - docs/REFACTORING_SUMMARY.md - Test results and recommendations
 * - docs/FFI_GUIDE.md - How to modify bindings
 */

#include "logger_bridge.hpp"
#include <NvOnnxParser.h>
#include <cstdint>
#include <cstring>

//==============================================================================
// SECTION 1: LOGGER BRIDGE (NECESSARY - Virtual Methods)
//==============================================================================
// Rust cannot implement C++ virtual classes, so we need this C++ class
// to forward ILogger::log() calls back to Rust via function pointer callbacks.

// C++ implementation of ILogger that bridges to Rust
class RustLoggerImpl : public nvinfer1::ILogger {
public:
  RustLoggerImpl(RustLogCallback callback, void *user_data)
      : callback_(callback), user_data_(user_data) {}

  void log(int32_t severity, const char *msg) noexcept override {
    if (callback_) {
      callback_(user_data_, static_cast<int32_t>(severity), msg);
    }
  }

private:
  RustLogCallback callback_;
  void *user_data_;
};

// Opaque struct that holds the logger implementation
struct RustLoggerBridge {
  RustLoggerImpl *impl;
};

extern "C" {

RustLoggerBridge *create_rust_logger_bridge(RustLogCallback callback,
                                            void *user_data) {
  if (!callback) {
    return nullptr;
  }

  try {
    auto *bridge = new RustLoggerBridge();
    bridge->impl = new RustLoggerImpl(callback, user_data);
    return bridge;
  } catch (...) {
    return nullptr;
  }
}

void destroy_rust_logger_bridge(RustLoggerBridge *logger) {
  if (logger) {
    delete logger->impl;
    delete logger;
  }
}

nvinfer1::ILogger *get_logger_interface(RustLoggerBridge *logger) {
  return logger ? logger->impl : nullptr;
}

//==============================================================================
// SECTION 2: FACTORY FUNCTIONS (NECESSARY - Reference Parameters)
//==============================================================================
// These functions take `ILogger&` references which autocxx struggles with.
// Simpler to keep these thin wrappers than to work around autocxx limitations.

// Factory functions for TensorRT
#ifdef TRTX_LINK_TENSORRT_RTX
void *create_infer_builder(void *logger) {
  if (!logger) {
    return nullptr;
  }
  try {
    auto *ilogger = static_cast<nvinfer1::ILogger *>(logger);
    return nvinfer1::createInferBuilder(*ilogger);
  } catch (...) {
    return nullptr;
  }
}

void *create_infer_runtime(void *logger) {
  if (!logger) {
    return nullptr;
  }
  try {
    auto *ilogger = static_cast<nvinfer1::ILogger *>(logger);
    return nvinfer1::createInferRuntime(*ilogger);
  } catch (...) {
    return nullptr;
  }
}
#endif

#ifdef TRTX_LINK_TENSORRT_ONNXPARSER
// ONNX Parser factory function
void *create_onnx_parser(void *network, void *logger) {
  if (!network || !logger) {
    return nullptr;
  }
  try {
    auto *inetwork = static_cast<nvinfer1::INetworkDefinition *>(network);
    auto *ilogger = static_cast<nvinfer1::ILogger *>(logger);
    return nvonnxparser::createParser(*inetwork, *ilogger);
  } catch (...) {
    return nullptr;
  }
}
#endif

bool parser_parse(void *parser, const void *data, size_t size) {
  if (!parser || !data)
    return false;
  try {
    auto *iparser = static_cast<nvonnxparser::IParser *>(parser);
    return iparser->parse(data, size);
  } catch (...) {
    return false;
  }
}

int32_t parser_get_nb_errors(void *parser) {
  if (!parser)
    return 0;
  try {
    auto *iparser = static_cast<nvonnxparser::IParser *>(parser);
    return iparser->getNbErrors();
  } catch (...) {
    return 0;
  }
}

void *parser_get_error(void *parser, int32_t index) {
  if (!parser)
    return nullptr;
  try {
    auto *iparser = static_cast<nvonnxparser::IParser *>(parser);
    return const_cast<nvonnxparser::IParserError *>(iparser->getError(index));
  } catch (...) {
    return nullptr;
  }
}

const char *parser_error_desc(void *error) {
  if (!error)
    return nullptr;
  try {
    auto *ierror = static_cast<nvonnxparser::IParserError *>(error);
    return ierror->desc();
  } catch (...) {
    return nullptr;
  }
}

void* network_add_concatenation(void* network, void** inputs, int32_t nb_inputs) {
    if (!network || !inputs || nb_inputs <= 0) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        std::vector<nvinfer1::ITensor*> tensors;
        tensors.reserve(nb_inputs);
        for (int32_t i = 0; i < nb_inputs; ++i) {
            tensors.push_back(static_cast<nvinfer1::ITensor*>(inputs[i]));
        }
        auto* layer = inetwork->addConcatenation(tensors.data(), nb_inputs);
        return layer; // Return layer, not output tensor
    } catch (...) {
        return nullptr;
    }
}

uint32_t get_tensorrt_version() { return NV_TENSORRT_VERSION; }

} // extern "C"
