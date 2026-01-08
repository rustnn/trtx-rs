// Mock implementations for development without TensorRT-RTX
// These are stubs that allow compilation and basic testing

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

// Mock handles (just use integers)
typedef struct { int dummy; } TrtxLogger;
typedef struct { int dummy; } TrtxBuilder;
typedef struct { int dummy; } TrtxBuilderConfig;
typedef struct { int dummy; } TrtxNetworkDefinition;
typedef struct { int dummy; } TrtxRuntime;
typedef struct { int dummy; } TrtxCudaEngine;
typedef struct { int dummy; } TrtxExecutionContext;

// Mock implementations - all return success

int32_t trtx_logger_create(
    void* callback,
    void* user_data,
    TrtxLogger** out_logger,
    char* error_msg,
    size_t error_msg_len
) {
    (void)callback;
    (void)user_data;
    (void)error_msg;
    (void)error_msg_len;
    *out_logger = malloc(sizeof(TrtxLogger));
    return 0; // TRTX_SUCCESS
}

void trtx_logger_destroy(TrtxLogger* logger) {
    free(logger);
}

int32_t trtx_builder_create(
    TrtxLogger* logger,
    TrtxBuilder** out_builder,
    char* error_msg,
    size_t error_msg_len
) {
    (void)logger;
    (void)error_msg;
    (void)error_msg_len;
    *out_builder = malloc(sizeof(TrtxBuilder));
    return 0;
}

void trtx_builder_destroy(TrtxBuilder* builder) {
    free(builder);
}

int32_t trtx_builder_create_network(
    TrtxBuilder* builder,
    uint32_t flags,
    TrtxNetworkDefinition** out_network,
    char* error_msg,
    size_t error_msg_len
) {
    (void)builder;
    (void)flags;
    (void)error_msg;
    (void)error_msg_len;
    *out_network = malloc(sizeof(TrtxNetworkDefinition));
    return 0;
}

int32_t trtx_builder_create_builder_config(
    TrtxBuilder* builder,
    TrtxBuilderConfig** out_config,
    char* error_msg,
    size_t error_msg_len
) {
    (void)builder;
    (void)error_msg;
    (void)error_msg_len;
    *out_config = malloc(sizeof(TrtxBuilderConfig));
    return 0;
}

int32_t trtx_builder_build_serialized_network(
    TrtxBuilder* builder,
    TrtxNetworkDefinition* network,
    TrtxBuilderConfig* config,
    void** out_data,
    size_t* out_size,
    char* error_msg,
    size_t error_msg_len
) {
    (void)builder;
    (void)network;
    (void)config;
    (void)error_msg;
    (void)error_msg_len;
    // Return a small dummy buffer
    *out_size = 16;
    *out_data = malloc(16);
    memset(*out_data, 0, 16);
    return 0;
}

void trtx_builder_config_destroy(TrtxBuilderConfig* config) {
    free(config);
}

int32_t trtx_builder_config_set_memory_pool_limit(
    TrtxBuilderConfig* config,
    int32_t pool_type,
    size_t pool_size,
    char* error_msg,
    size_t error_msg_len
) {
    (void)config;
    (void)pool_type;
    (void)pool_size;
    (void)error_msg;
    (void)error_msg_len;
    return 0;
}

void trtx_network_destroy(TrtxNetworkDefinition* network) {
    free(network);
}

int32_t trtx_runtime_create(
    TrtxLogger* logger,
    TrtxRuntime** out_runtime,
    char* error_msg,
    size_t error_msg_len
) {
    (void)logger;
    (void)error_msg;
    (void)error_msg_len;
    *out_runtime = malloc(sizeof(TrtxRuntime));
    return 0;
}

void trtx_runtime_destroy(TrtxRuntime* runtime) {
    free(runtime);
}

int32_t trtx_runtime_deserialize_cuda_engine(
    TrtxRuntime* runtime,
    const void* data,
    size_t size,
    TrtxCudaEngine** out_engine,
    char* error_msg,
    size_t error_msg_len
) {
    (void)runtime;
    (void)data;
    (void)size;
    (void)error_msg;
    (void)error_msg_len;
    *out_engine = malloc(sizeof(TrtxCudaEngine));
    return 0;
}

void trtx_cuda_engine_destroy(TrtxCudaEngine* engine) {
    free(engine);
}

int32_t trtx_cuda_engine_create_execution_context(
    TrtxCudaEngine* engine,
    TrtxExecutionContext** out_context,
    char* error_msg,
    size_t error_msg_len
) {
    (void)engine;
    (void)error_msg;
    (void)error_msg_len;
    *out_context = malloc(sizeof(TrtxExecutionContext));
    return 0;
}

int32_t trtx_cuda_engine_get_tensor_name(
    TrtxCudaEngine* engine,
    int32_t index,
    const char** out_name,
    char* error_msg,
    size_t error_msg_len
) {
    (void)engine;
    (void)error_msg;
    (void)error_msg_len;
    static const char* mock_names[] = {"input", "output"};
    if (index < 0 || index >= 2) {
        return 1; // TRTX_ERROR_INVALID_ARGUMENT
    }
    *out_name = mock_names[index];
    return 0;
}

int32_t trtx_cuda_engine_get_nb_io_tensors(
    TrtxCudaEngine* engine,
    int32_t* out_count
) {
    (void)engine;
    *out_count = 2; // Mock: 1 input, 1 output
    return 0;
}

void trtx_execution_context_destroy(TrtxExecutionContext* context) {
    free(context);
}

int32_t trtx_execution_context_set_tensor_address(
    TrtxExecutionContext* context,
    const char* tensor_name,
    void* data,
    char* error_msg,
    size_t error_msg_len
) {
    (void)context;
    (void)tensor_name;
    (void)data;
    (void)error_msg;
    (void)error_msg_len;
    return 0;
}

int32_t trtx_execution_context_enqueue_v3(
    TrtxExecutionContext* context,
    void* cuda_stream,
    char* error_msg,
    size_t error_msg_len
) {
    (void)context;
    (void)cuda_stream;
    (void)error_msg;
    (void)error_msg_len;
    return 0;
}

void trtx_free_buffer(void* buffer) {
    free(buffer);
}

// ONNX Parser mock implementations
typedef struct { int dummy; } TrtxOnnxParser;

int32_t trtx_onnx_parser_create(
    void* network,
    void* logger,
    TrtxOnnxParser** out_parser,
    char* error_msg,
    size_t error_msg_len
) {
    (void)network;
    (void)logger;
    (void)error_msg;
    (void)error_msg_len;
    *out_parser = malloc(sizeof(TrtxOnnxParser));
    return 0;
}

void trtx_onnx_parser_destroy(TrtxOnnxParser* parser) {
    free(parser);
}

int32_t trtx_onnx_parser_parse(
    TrtxOnnxParser* parser,
    const void* model_data,
    size_t model_size,
    char* error_msg,
    size_t error_msg_len
) {
    (void)parser;
    (void)model_data;
    (void)model_size;
    (void)error_msg;
    (void)error_msg_len;
    // Mock: always succeeds
    return 0;
}

// CUDA Memory Management mock implementations
int32_t trtx_cuda_malloc(
    void** ptr,
    size_t size,
    char* error_msg,
    size_t error_msg_len
) {
    (void)error_msg;
    (void)error_msg_len;
    *ptr = malloc(size);
    return *ptr ? 0 : 2; // 0 = success, 2 = out of memory
}

int32_t trtx_cuda_free(
    void* ptr,
    char* error_msg,
    size_t error_msg_len
) {
    (void)error_msg;
    (void)error_msg_len;
    free(ptr);
    return 0;
}

int32_t trtx_cuda_memcpy_host_to_device(
    void* dst,
    const void* src,
    size_t size,
    char* error_msg,
    size_t error_msg_len
) {
    (void)error_msg;
    (void)error_msg_len;
    memcpy(dst, src, size);
    return 0;
}

int32_t trtx_cuda_memcpy_device_to_host(
    void* dst,
    const void* src,
    size_t size,
    char* error_msg,
    size_t error_msg_len
) {
    (void)error_msg;
    (void)error_msg_len;
    memcpy(dst, src, size);
    return 0;
}

int32_t trtx_cuda_synchronize(
    char* error_msg,
    size_t error_msg_len
) {
    (void)error_msg;
    (void)error_msg_len;
    // Mock: nothing to synchronize
    return 0;
}

void* trtx_cuda_get_default_stream() {
    return NULL;
}
