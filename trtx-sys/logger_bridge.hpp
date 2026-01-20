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
void* builder_create_network_v2(void* builder, uint32_t flags);
void* builder_create_config(void* builder);
void* builder_build_serialized_network(void* builder, void* network, void* config, size_t* out_size);
void builder_config_set_memory_pool_limit(void* config, int32_t pool_type, size_t limit);

// Network methods
void* network_add_input(void* network, const char* name, int32_t data_type, const int32_t* dims, int32_t nb_dims);
bool network_mark_output(void* network, void* tensor);
int32_t network_get_nb_inputs(void* network);
int32_t network_get_nb_outputs(void* network);
void* network_get_input(void* network, int32_t index);
void* network_get_output(void* network, int32_t index);
void* network_add_convolution(void* network, void* input, int32_t nb_outputs, const int32_t* kernel_size, const void* weights, const void* bias);
void* network_add_activation(void* network, void* input, int32_t type);
void* network_add_pooling(void* network, void* input, int32_t type, const int32_t* window_size);
void* network_add_elementwise(void* network, void* input1, void* input2, int32_t op);
void* network_add_shuffle(void* network, void* input);
void* network_add_concatenation(void* network, void** inputs, int32_t nb_inputs);
void* network_add_matrix_multiply(void* network, void* input0, int32_t op0, void* input1, int32_t op1);
void* network_add_constant(void* network, const int32_t* dims, int32_t nb_dims, const void* weights);
void* network_add_softmax(void* network, void* input, uint32_t axes);
void* network_add_scale(void* network, void* input, int32_t mode, const void* shift, const void* scale, const void* power);
void* network_add_reduce(void* network, void* input, int32_t op, uint32_t axes, bool keep_dims);
void* network_add_slice(void* network, void* input, const int32_t* start, const int32_t* size, const int32_t* stride, int32_t nb_dims);
void* network_add_resize(void* network, void* input);
void* network_add_topk(void* network, void* input, int32_t op, int32_t k, uint32_t axes);
void* network_add_gather(void* network, void* data, void* indices, int32_t axis);
void* network_add_select(void* network, void* condition, void* then_input, void* else_input);
void* network_add_assertion(void* network, void* condition, const char* message);
void* network_add_loop(void* network);
void* network_add_if_conditional(void* network);

// Tensor methods
const char* tensor_get_name(void* tensor);
void tensor_set_name(void* tensor, const char* name);
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
int32_t engine_get_nb_io_tensors(void* engine);
const char* engine_get_tensor_name(void* engine, int32_t index);
void* engine_create_execution_context(void* engine);

// ExecutionContext methods
bool context_set_tensor_address(void* context, const char* name, void* data);
bool context_enqueue_v3(void* context, void* stream);

// Parser methods
bool parser_parse(void* parser, const void* data, size_t size);
int32_t parser_get_nb_errors(void* parser);
void* parser_get_error(void* parser, int32_t index);
const char* parser_error_desc(void* error);

// CUDA wrappers
int32_t cuda_malloc_wrapper(void** ptr, size_t size);
int32_t cuda_free_wrapper(void* ptr);
int32_t cuda_memcpy_wrapper(void* dst, const void* src, size_t count, int32_t kind);
int32_t cuda_device_synchronize_wrapper();
const char* cuda_get_error_string_wrapper(int32_t error);

#ifdef __cplusplus
}
#endif

#endif // TRTX_LOGGER_BRIDGE_H
