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
#include <NvInferRuntime.h>
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
void *create_infer_refitter(void *engine, void *logger) {
  if (!engine || !logger) {
    return nullptr;
  }
  try {
    auto *iengine = static_cast<nvinfer1::ICudaEngine *>(engine);
    auto *ilogger = static_cast<nvinfer1::ILogger *>(logger);
    return nvinfer1::createInferRefitter(*iengine, *ilogger);
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

// Refitter methods that use char const** (pointer-to-pointer); autocxx cannot
// bind these.
int32_t trtx_refitter_get_missing(void *refitter, int32_t size,
                                  char const **layer_names, int32_t *roles) {
  if (!refitter || !layer_names || !roles)
    return 0;
  try {
    auto *ir = static_cast<nvinfer1::IRefitter *>(refitter);
    return ir->getMissing(size, layer_names,
                          reinterpret_cast<nvinfer1::WeightsRole *>(roles));
  } catch (...) {
    return 0;
  }
}

int32_t trtx_refitter_get_all(void *refitter, int32_t size,
                              char const **layer_names, int32_t *roles) {
  if (!refitter || !layer_names || !roles)
    return 0;
  try {
    auto *ir = static_cast<nvinfer1::IRefitter *>(refitter);
    return ir->getAll(size, layer_names,
                      reinterpret_cast<nvinfer1::WeightsRole *>(roles));
  } catch (...) {
    return 0;
  }
}

int32_t trtx_refitter_get_missing_weights(void *refitter, int32_t size,
                                          char const **weights_names) {
  if (!refitter || !weights_names)
    return 0;
  try {
    auto *ir = static_cast<nvinfer1::IRefitter *>(refitter);
    return ir->getMissingWeights(size, weights_names);
  } catch (...) {
    return 0;
  }
}

int32_t trtx_refitter_get_all_weights(void *refitter, int32_t size,
                                      char const **weights_names) {
  if (!refitter || !weights_names)
    return 0;
  try {
    auto *ir = static_cast<nvinfer1::IRefitter *>(refitter);
    return ir->getAllWeights(size, weights_names);
  } catch (...) {
    return 0;
  }
}

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

void *network_add_concatenation(void *network, void **inputs,
                                int32_t nb_inputs) {
  if (!network || !inputs || nb_inputs <= 0)
    return nullptr;
  try {
    auto *inetwork = static_cast<nvinfer1::INetworkDefinition *>(network);
    std::vector<nvinfer1::ITensor *> tensors;
    tensors.reserve(nb_inputs);
    for (int32_t i = 0; i < nb_inputs; ++i) {
      tensors.push_back(static_cast<nvinfer1::ITensor *>(inputs[i]));
    }
    auto *layer = inetwork->addConcatenation(tensors.data(), nb_inputs);
    return layer; // Return layer, not output tensor
  } catch (...) {
    return nullptr;
  }
}

uint32_t get_tensorrt_version() { return NV_TENSORRT_VERSION; }

namespace nvinfer1 {
class ProgressMonitor : public IProgressMonitor {
public:
  ProgressMonitor(void *self, void *phaseStart, void *stepComplete,
                  void *phaseFinish)
      : self(self), m_phaseStart((decltype(m_phaseStart))phaseStart),
        m_stepComplete((decltype(m_stepComplete))stepComplete),
        m_phaseFinish((decltype(m_phaseFinish))phaseFinish) {}
  ~ProgressMonitor() = default;
  void *self;
  void (*m_phaseStart)(void *self, char const *phaseName,
                       char const *parentPhase, int32_t nbSteps);
  bool (*m_stepComplete)(void *self, char const *phaseName, int32_t step);
  void (*m_phaseFinish)(void *self, char const *phaseName);

  void phaseStart(char const *phaseName, char const *parentPhase,
                  int32_t nbSteps) noexcept override {
    m_phaseStart(self, phaseName, parentPhase, nbSteps);
  };
  bool stepComplete(char const *phaseName, int32_t step) noexcept override {
    return m_stepComplete(self, phaseName, step);
  };
  void phaseFinish(char const *phaseName) noexcept override {
    m_phaseFinish(self, phaseName);
  };
};
} // namespace nvinfer1

void *trtx_create_progress_monitor(void *self, void *phaseStart,
                                   void *stepComplete, void *phaseFinish) {
  try {
    return new nvinfer1::ProgressMonitor(self, phaseStart, stepComplete,
                                         phaseFinish);
  } catch (...) {
    return nullptr;
  }
}
void trtx_destroy_progress_monitor(void *self) {
  delete (nvinfer1::ProgressMonitor *)(self);
}

//==============================================================================
// ErrorRecorder subclass (bridge to Rust RecordError)
//==============================================================================
namespace nvinfer1 {
class ErrorRecorderSubclass : public IErrorRecorder {
public:
  using ErrorCode = nvinfer1::ErrorCode;
  ErrorRecorderSubclass(void *self, int32_t (*getNbErrors)(void *),
                        int32_t (*getErrorCode)(void *, int32_t),
                        void (*getErrorDesc)(void *, int32_t, char *, size_t),
                        bool (*hasOverflowed)(void *), void (*clear)(void *),
                        bool (*reportError)(void *, int32_t, char const *),
                        int32_t (*incRefCount)(void *),
                        int32_t (*decRefCount)(void *))
      : self(self), m_getNbErrors(getNbErrors), m_getErrorCode(getErrorCode),
        m_getErrorDesc(getErrorDesc), m_hasOverflowed(hasOverflowed),
        m_clear(clear), m_reportError(reportError), m_incRefCount(incRefCount),
        m_decRefCount(decRefCount) {}
  ~ErrorRecorderSubclass() = default;

  void *self;
  int32_t (*m_getNbErrors)(void *);
  int32_t (*m_getErrorCode)(void *, int32_t);
  void (*m_getErrorDesc)(void *, int32_t, char *, size_t);
  bool (*m_hasOverflowed)(void *);
  void (*m_clear)(void *);
  bool (*m_reportError)(void *, int32_t, char const *);
  int32_t (*m_incRefCount)(void *);
  int32_t (*m_decRefCount)(void *);

  mutable std::string m_lastDesc;

  int32_t getNbErrors() const noexcept override { return m_getNbErrors(self); }
  int32_t getErrorCode(int32_t errorIdx) const noexcept override {
    return m_getErrorCode(self, errorIdx);
  }
  ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override {
    char buf[128];
    m_getErrorDesc(self, errorIdx, buf, sizeof(buf));
    m_lastDesc = buf;
    return m_lastDesc.c_str();
  }
  bool hasOverflowed() const noexcept override { return m_hasOverflowed(self); }
  void clear() noexcept override { m_clear(self); }
  bool reportError(int32_t val, ErrorDesc desc) noexcept override {
    return m_reportError(self, val, desc);
  }
  RefCount incRefCount() noexcept override { return m_incRefCount(self); }
  RefCount decRefCount() noexcept override { return m_decRefCount(self); }
};
} // namespace nvinfer1

void *trtx_create_error_recorder(void *self, void *getNbErrors,
                                 void *getErrorCode, void *getErrorDesc,
                                 void *hasOverflowed, void *clear,
                                 void *reportError, void *incRefCount,
                                 void *decRefCount) {
  try {
    return new nvinfer1::ErrorRecorderSubclass(
        self, (int32_t (*)(void *))getNbErrors,
        (int32_t (*)(void *, int32_t))getErrorCode,
        (void (*)(void *, int32_t, char *, size_t))getErrorDesc,
        (bool (*)(void *))hasOverflowed, (void (*)(void *))clear,
        (bool (*)(void *, int32_t, char const *))reportError,
        (int32_t (*)(void *))incRefCount, (int32_t (*)(void *))decRefCount);
  } catch (...) {
    return nullptr;
  }
}
void trtx_destroy_error_recorder(void *obj) {
  delete static_cast<nvinfer1::ErrorRecorderSubclass *>(obj);
}

//==============================================================================
// GpuAllocator subclass (bridge to Rust AllocateGpu)
//==============================================================================
namespace nvinfer1 {
class GpuAllocatorSubclass : public IGpuAllocator {
public:
  GpuAllocatorSubclass(void *self,
                       void *(*allocateAsync)(void *, uint64_t, uint64_t,
                                              uint32_t, void *),
                       void *(*reallocate)(void *, void *, uint64_t, uint64_t),
                       bool (*deallocateAsync)(void *, void *, void *))
      : self(self), m_allocateAsync((decltype(m_allocateAsync))allocateAsync),
        m_reallocate((decltype(m_reallocate))reallocate),
        m_deallocateAsync((decltype(m_deallocateAsync))deallocateAsync) {}
  ~GpuAllocatorSubclass() = default;

  void *self;
  void *(*m_allocateAsync)(void *, uint64_t, uint64_t, uint32_t, void *);
  void *(*m_reallocate)(void *, void *, uint64_t, uint64_t);
  bool (*m_deallocateAsync)(void *, void *, void *);

  void *allocate(uint64_t size, uint64_t alignment,
                 AllocatorFlags flags) noexcept override {
    return m_allocateAsync(self, size, alignment, static_cast<uint32_t>(flags),
                           nullptr);
  }
  void *reallocate(void *baseAddr, uint64_t alignment,
                   uint64_t newSize) noexcept override {
    return m_reallocate(self, baseAddr, alignment, newSize);
  }
  bool deallocate(void *memory) noexcept override {
    return m_deallocateAsync(self, memory, nullptr);
  }
  void *allocateAsync(uint64_t size, uint64_t alignment, AllocatorFlags flags,
                      cudaStream_t stream) noexcept override {
    return m_allocateAsync(self, size, alignment, static_cast<uint32_t>(flags),
                           stream);
  }
  bool deallocateAsync(void *memory, cudaStream_t stream) noexcept override {
    return m_deallocateAsync(self, memory, stream);
  }
};
} // namespace nvinfer1

void *trtx_create_gpu_allocator(void *self, void *allocateAsync,
                                void *reallocate, void *deallocateAsync) {

  try {
    return new nvinfer1::GpuAllocatorSubclass(
        self,
        (void *(*)(void *, uint64_t, uint64_t, uint32_t, void *))allocateAsync,
        (void *(*)(void *, void *, uint64_t, uint64_t))reallocate,
        (bool (*)(void *, void *, void *))deallocateAsync);
  } catch (...) {
    return nullptr;
  }
}
void trtx_destroy_gpu_allocator(void *obj) {
  delete static_cast<nvinfer1::GpuAllocatorSubclass *>(obj);
}
}

namespace nvinfer1 {
class DebugListener : public IDebugListener {
public:
  DebugListener(void *self,
                bool (*processDebugTensor)(void *self, void const *addr,
                                           TensorLocation location,
                                           DataType type, Dims const *shape,
                                           char const *name,
                                           cudaStream_t stream))
      : self(self), m_processDebugTensor(
                        (decltype(m_processDebugTensor))processDebugTensor) {}
  ~DebugListener() = default;

  void *self;
  bool (*m_processDebugTensor)(void *self, void const *addr,
                               TensorLocation location, DataType type,
                               Dims const *shape, char const *name,
                               cudaStream_t stream);

  bool processDebugTensor(void const *addr, TensorLocation location,
                          DataType type, Dims const &shape, char const *name,
                          cudaStream_t stream) noexcept override {
    return m_processDebugTensor(self, addr, location, type, &shape, name,
                                stream);
  };
};

extern "C" {
void *trtx_create_debug_listener(
    nvinfer1::IDebugListener *self,
    bool (*processDebugTensor)(void *self, void const *addr,
                               TensorLocation location, DataType type,
                               Dims const *shape, char const *name,
                               cudaStream_t stream)) {
  return new DebugListener(self, processDebugTensor);
}
void trtx_destroy_debug_listener(nvinfer1::IDebugListener *self) {
  delete self;
}
}
} // namespace nvinfer1
