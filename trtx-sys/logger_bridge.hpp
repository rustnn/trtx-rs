#ifndef TRTX_LOGGER_BRIDGE_H
#define TRTX_LOGGER_BRIDGE_H

#include <NvInfer.h>

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
void* network_add_input(void* network, const char* name, int32_t data_type, const int32_t* dims, int32_t nb_dims);
void* network_add_convolution(void* network, void* input, int32_t nb_outputs, const int32_t* kernel_size, const void* weights, const void* bias);
void* network_add_pooling(void* network, void* input, int32_t type, const int32_t* window_size);
void* network_add_concatenation(void* network, void** inputs, int32_t nb_inputs);
void* network_add_constant(void* network, const int32_t* dims, int32_t nb_dims, const void* weights, int32_t data_type, int64_t count);
void* network_add_scale(void* network, void* input, int32_t mode, const void* shift, const void* scale, const void* power);
void* network_add_slice(void* network, void* input, const int32_t* start, const int32_t* size, const int32_t* stride, int32_t nb_dims);
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
