#ifndef TRTX_LOGGER_BRIDGE_H
#define TRTX_LOGGER_BRIDGE_H

#include <cstddef>
namespace std {
typedef std::size_t size_t;
}
#include <NvInfer.h>

#include <type_traits>// This will cause a compiler error if Weights isn't a POD
static_assert(std::is_standard_layout<nvinfer1::Weights>::value, "Weights must be standard layout");
static_assert(std::is_trivial<nvinfer1::Weights>::value, "Weights must be trivial");

#ifdef __cplusplus
extern "C" {
#endif

// Rust callback function type
typedef void (*RustLogCallback)(void* user_data, int32_t severity, const char* msg);

// Opaque logger type for C interface
typedef struct RustLoggerBridge RustLoggerBridge;

// Create a logger bridge that calls back into Rust
RustLoggerBridge* create_rust_logger_bridge(RustLogCallback callback, void* user_data);

// Destroy the logger bridge
void destroy_rust_logger_bridge(RustLoggerBridge* logger);

// Get the ILogger pointer (for use with TensorRT C++ API)
nvinfer1::ILogger* get_logger_interface(RustLoggerBridge* logger);

// Factory functions for TensorRT (return raw pointers)
void* create_infer_builder(void* logger);
void* create_infer_runtime(void* logger);

// ONNX Parser factory function
void* create_onnx_parser(void* network, void* logger);

// Builder methods
void* builder_build_serialized_network(void* builder, void* network, void* config, size_t* out_size);
void builder_config_set_memory_pool_limit(void* config, int32_t pool_type, size_t limit);

// Network methods
// network_add_convolution - REMOVED - Using direct autocxx
void* network_add_concatenation(void* network, void** inputs, int32_t nb_inputs);
// network_add_constant - REMOVED - Using direct autocxx
// network_add_scale - REMOVED - Using direct autocxx
void* network_add_assertion(void* network, void* condition, const char* message);
void* network_add_loop(void* network);
void* network_add_if_conditional(void* network);

// Tensor methods
void* tensor_get_dimensions(void* tensor, int32_t* dims, int32_t* nb_dims);
int32_t tensor_get_type(void* tensor);

// Destruction methods
void delete_builder(void* builder);
void delete_network(void* network);
void delete_config(void* config);
void delete_runtime(void* runtime);
void delete_engine(void* engine);
void delete_context(void* context);
void delete_parser(void* parser);

// Runtime methods
void* runtime_deserialize_cuda_engine(void* runtime, const void* data, size_t size);

// Engine methods

// ExecutionContext methods

// Parser methods
bool parser_parse(void* parser, const void* data, size_t size);
int32_t parser_get_nb_errors(void* parser);
void* parser_get_error(void* parser, int32_t index);
const char* parser_error_desc(void* error);

#ifdef __cplusplus
}
#endif

#endif // TRTX_LOGGER_BRIDGE_H
