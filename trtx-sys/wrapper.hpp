#ifndef TRTX_WRAPPER_H
#define TRTX_WRAPPER_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
#define TRTX_SUCCESS 0
#define TRTX_ERROR_INVALID_ARGUMENT 1
#define TRTX_ERROR_OUT_OF_MEMORY 2
#define TRTX_ERROR_RUNTIME_ERROR 3
#define TRTX_ERROR_CUDA_ERROR 4
#define TRTX_ERROR_UNKNOWN 99

// Logger severity levels (matching nvinfer1::ILogger::Severity)
typedef enum : int32_t {
    TRTX_SEVERITY_INTERNAL_ERROR = 0,
    TRTX_SEVERITY_ERROR = 1,
    TRTX_SEVERITY_WARNING = 2,
    TRTX_SEVERITY_INFO = 3,
    TRTX_SEVERITY_VERBOSE = 4
} TrtxLoggerSeverity;

// Opaque types
typedef struct TrtxLogger TrtxLogger;
typedef struct TrtxBuilder TrtxBuilder;
typedef struct TrtxBuilderConfig TrtxBuilderConfig;
typedef struct TrtxNetworkDefinition TrtxNetworkDefinition;
typedef struct TrtxRuntime TrtxRuntime;
typedef struct TrtxCudaEngine TrtxCudaEngine;
typedef struct TrtxExecutionContext TrtxExecutionContext;

// Logger callback type
typedef void (*TrtxLoggerCallback)(void* user_data, TrtxLoggerSeverity severity, const char* msg);

// Logger functions
int32_t trtx_logger_create(
    TrtxLoggerCallback callback,
    void* user_data,
    TrtxLogger** out_logger,
    char* error_msg,
    size_t error_msg_len
);

void trtx_logger_destroy(TrtxLogger* logger);

// Builder functions
int32_t trtx_builder_create(
    TrtxLogger* logger,
    TrtxBuilder** out_builder,
    char* error_msg,
    size_t error_msg_len
);

void trtx_builder_destroy(TrtxBuilder* builder);

int32_t trtx_builder_create_network(
    TrtxBuilder* builder,
    uint32_t flags,
    TrtxNetworkDefinition** out_network,
    char* error_msg,
    size_t error_msg_len
);

int32_t trtx_builder_create_builder_config(
    TrtxBuilder* builder,
    TrtxBuilderConfig** out_config,
    char* error_msg,
    size_t error_msg_len
);

int32_t trtx_builder_build_serialized_network(
    TrtxBuilder* builder,
    TrtxNetworkDefinition* network,
    TrtxBuilderConfig* config,
    void** out_data,
    size_t* out_size,
    char* error_msg,
    size_t error_msg_len
);

// BuilderConfig functions
void trtx_builder_config_destroy(TrtxBuilderConfig* config);

int32_t trtx_builder_config_set_memory_pool_limit(
    TrtxBuilderConfig* config,
    int32_t pool_type,
    size_t pool_size,
    char* error_msg,
    size_t error_msg_len
);

// NetworkDefinition functions
void trtx_network_destroy(TrtxNetworkDefinition* network);

// Runtime functions
int32_t trtx_runtime_create(
    TrtxLogger* logger,
    TrtxRuntime** out_runtime,
    char* error_msg,
    size_t error_msg_len
);

void trtx_runtime_destroy(TrtxRuntime* runtime);

int32_t trtx_runtime_deserialize_cuda_engine(
    TrtxRuntime* runtime,
    const void* data,
    size_t size,
    TrtxCudaEngine** out_engine,
    char* error_msg,
    size_t error_msg_len
);

// CudaEngine functions
void trtx_cuda_engine_destroy(TrtxCudaEngine* engine);

int32_t trtx_cuda_engine_create_execution_context(
    TrtxCudaEngine* engine,
    TrtxExecutionContext** out_context,
    char* error_msg,
    size_t error_msg_len
);

int32_t trtx_cuda_engine_get_tensor_name(
    TrtxCudaEngine* engine,
    int32_t index,
    const char** out_name,
    char* error_msg,
    size_t error_msg_len
);

int32_t trtx_cuda_engine_get_nb_io_tensors(
    TrtxCudaEngine* engine,
    int32_t* out_count
);

// ExecutionContext functions
void trtx_execution_context_destroy(TrtxExecutionContext* context);

int32_t trtx_execution_context_set_tensor_address(
    TrtxExecutionContext* context,
    const char* tensor_name,
    void* data,
    char* error_msg,
    size_t error_msg_len
);

int32_t trtx_execution_context_enqueue_v3(
    TrtxExecutionContext* context,
    void* cuda_stream,
    char* error_msg,
    size_t error_msg_len
);

// Utility functions
void trtx_free_buffer(void* buffer);

// ONNX Parser functions
typedef struct TrtxOnnxParser TrtxOnnxParser;

int32_t trtx_onnx_parser_create(
    TrtxNetworkDefinition* network,
    TrtxLogger* logger,
    TrtxOnnxParser** out_parser,
    char* error_msg,
    size_t error_msg_len
);

void trtx_onnx_parser_destroy(TrtxOnnxParser* parser);

int32_t trtx_onnx_parser_parse(
    TrtxOnnxParser* parser,
    const void* model_data,
    size_t model_size,
    char* error_msg,
    size_t error_msg_len
);

// CUDA Memory Management functions
int32_t trtx_cuda_malloc(
    void** ptr,
    size_t size,
    char* error_msg,
    size_t error_msg_len
);

int32_t trtx_cuda_free(
    void* ptr,
    char* error_msg,
    size_t error_msg_len
);

int32_t trtx_cuda_memcpy_host_to_device(
    void* dst,
    const void* src,
    size_t size,
    char* error_msg,
    size_t error_msg_len
);

int32_t trtx_cuda_memcpy_device_to_host(
    void* dst,
    const void* src,
    size_t size,
    char* error_msg,
    size_t error_msg_len
);

int32_t trtx_cuda_synchronize(
    char* error_msg,
    size_t error_msg_len
);

// Helper function to get default CUDA stream (returns NULL for default stream)
void* trtx_cuda_get_default_stream();

#ifdef __cplusplus
}
#endif

#endif // TRTX_WRAPPER_H
