#ifndef TRTX_LOGGER_BRIDGE_H
#define TRTX_LOGGER_BRIDGE_H

#include <NvInfer.h>
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Rust callback function type
typedef void (*RustLogCallback)(void *user_data, int32_t severity,
                                const char *msg);

// Opaque logger type for C interface
typedef struct RustLoggerBridge RustLoggerBridge;

// Create a logger bridge that calls back into Rust
RustLoggerBridge *create_rust_logger_bridge(RustLogCallback callback,
                                            void *user_data);

// Destroy the logger bridge
void destroy_rust_logger_bridge(RustLoggerBridge *logger);

// Get the ILogger pointer (for use with TensorRT C++ API)
nvinfer1::ILogger *get_logger_interface(RustLoggerBridge *logger);

#ifdef __cplusplus
}
#endif

#endif // TRTX_LOGGER_BRIDGE_H
