#include "logger_bridge.hpp"
#include <NvOnnxParser.h>
#include <cstring>

// C++ implementation of ILogger that bridges to Rust
class RustLoggerImpl : public nvinfer1::ILogger {
public:
    RustLoggerImpl(RustLogCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (callback_) {
            callback_(user_data_, static_cast<int32_t>(severity), msg);
        }
    }

private:
    RustLogCallback callback_;
    void* user_data_;
};

// Opaque struct that holds the logger implementation
struct RustLoggerBridge {
    RustLoggerImpl* impl;
};

extern "C" {

RustLoggerBridge* create_rust_logger_bridge(RustLogCallback callback, void* user_data) {
    if (!callback) {
        return nullptr;
    }
    
    try {
        auto* bridge = new RustLoggerBridge();
        bridge->impl = new RustLoggerImpl(callback, user_data);
        return bridge;
    } catch (...) {
        return nullptr;
    }
}

void destroy_rust_logger_bridge(RustLoggerBridge* logger) {
    if (logger) {
        delete logger->impl;
        delete logger;
    }
}

nvinfer1::ILogger* get_logger_interface(RustLoggerBridge* logger) {
    return logger ? logger->impl : nullptr;
}

// Factory functions for TensorRT
void* create_infer_builder(void* logger) {
    if (!logger) {
        return nullptr;
    }
    try {
        auto* ilogger = static_cast<nvinfer1::ILogger*>(logger);
        return nvinfer1::createInferBuilder(*ilogger);
    } catch (...) {
        return nullptr;
    }
}

void* create_infer_runtime(void* logger) {
    if (!logger) {
        return nullptr;
    }
    try {
        auto* ilogger = static_cast<nvinfer1::ILogger*>(logger);
        return nvinfer1::createInferRuntime(*ilogger);
    } catch (...) {
        return nullptr;
    }
}

// ONNX Parser factory function
void* create_onnx_parser(void* network, void* logger) {
    if (!network || !logger) {
        return nullptr;
    }
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* ilogger = static_cast<nvinfer1::ILogger*>(logger);
        return nvonnxparser::createParser(*inetwork, *ilogger);
    } catch (...) {
        return nullptr;
    }
}

// Builder methods
void* builder_create_network_v2(void* builder, uint32_t flags) {
    if (!builder) return nullptr;
    try {
        auto* ibuilder = static_cast<nvinfer1::IBuilder*>(builder);
        return ibuilder->createNetworkV2(flags);
    } catch (...) {
        return nullptr;
    }
}

void* builder_create_config(void* builder) {
    if (!builder) return nullptr;
    try {
        auto* ibuilder = static_cast<nvinfer1::IBuilder*>(builder);
        return ibuilder->createBuilderConfig();
    } catch (...) {
        return nullptr;
    }
}

void builder_config_set_memory_pool_limit(void* config, int32_t pool_type, size_t limit) {
    if (!config) return;
    try {
        auto* iconfig = static_cast<nvinfer1::IBuilderConfig*>(config);
        iconfig->setMemoryPoolLimit(static_cast<nvinfer1::MemoryPoolType>(pool_type), limit);
    } catch (...) {
        // Ignore errors
    }
}

// Network methods
void* network_add_input(void* network, const char* name, int32_t data_type, const int32_t* dims, int32_t nb_dims) {
    if (!network || !name || !dims) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        nvinfer1::Dims dimensions;
        dimensions.nbDims = nb_dims;
        for (int32_t i = 0; i < nb_dims && i < nvinfer1::Dims::MAX_DIMS; ++i) {
            dimensions.d[i] = dims[i];
        }
        return inetwork->addInput(name, static_cast<nvinfer1::DataType>(data_type), dimensions);
    } catch (...) {
        return nullptr;
    }
}

bool network_mark_output(void* network, void* tensor) {
    if (!network || !tensor) return false;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(tensor);
        inetwork->markOutput(*itensor);
        return true;
    } catch (...) {
        return false;
    }
}

int32_t network_get_nb_inputs(void* network) {
    if (!network) return 0;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        return inetwork->getNbInputs();
    } catch (...) {
        return 0;
    }
}

int32_t network_get_nb_outputs(void* network) {
    if (!network) return 0;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        return inetwork->getNbOutputs();
    } catch (...) {
        return 0;
    }
}

void* network_get_input(void* network, int32_t index) {
    if (!network) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        return inetwork->getInput(index);
    } catch (...) {
        return nullptr;
    }
}

void* network_get_output(void* network, int32_t index) {
    if (!network) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        return inetwork->getOutput(index);
    } catch (...) {
        return nullptr;
    }
}

void* network_add_convolution(void* network, void* input, int32_t nb_outputs, const int32_t* kernel_size, const void* weights, const void* bias) {
    if (!network || !input || !kernel_size) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(input);
        nvinfer1::Dims dims;
        dims.nbDims = 2; // Assuming 2D kernel
        dims.d[0] = kernel_size[0];
        dims.d[1] = kernel_size[1];
        
        nvinfer1::Weights w{static_cast<nvinfer1::DataType>(0), weights, 0};
        nvinfer1::Weights b{static_cast<nvinfer1::DataType>(0), bias, 0};
        
        auto* layer = inetwork->addConvolutionNd(*itensor, nb_outputs, dims, w, b);
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_activation(void* network, void* input, int32_t type) {
    if (!network || !input) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(input);
        auto* layer = inetwork->addActivation(*itensor, static_cast<nvinfer1::ActivationType>(type));
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_pooling(void* network, void* input, int32_t type, const int32_t* window_size) {
    if (!network || !input || !window_size) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(input);
        nvinfer1::Dims dims;
        dims.nbDims = 2; // Assuming 2D pooling
        dims.d[0] = window_size[0];
        dims.d[1] = window_size[1];
        auto* layer = inetwork->addPoolingNd(*itensor, static_cast<nvinfer1::PoolingType>(type), dims);
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_matrix_multiply(void* network, void* input0, int32_t op0, void* input1, int32_t op1) {
    if (!network || !input0 || !input1) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor0 = static_cast<nvinfer1::ITensor*>(input0);
        auto* itensor1 = static_cast<nvinfer1::ITensor*>(input1);
        auto* layer = inetwork->addMatrixMultiply(
            *itensor0, static_cast<nvinfer1::MatrixOperation>(op0),
            *itensor1, static_cast<nvinfer1::MatrixOperation>(op1)
        );
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_constant(void* network, const int32_t* dims, int32_t nb_dims, const void* weights) {
    if (!network || !dims || !weights) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        nvinfer1::Dims dimensions;
        dimensions.nbDims = nb_dims;
        for (int32_t i = 0; i < nb_dims && i < nvinfer1::Dims::MAX_DIMS; ++i) {
            dimensions.d[i] = dims[i];
        }
        nvinfer1::Weights w{static_cast<nvinfer1::DataType>(0), weights, 0};
        auto* layer = inetwork->addConstant(dimensions, w);
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_elementwise(void* network, void* input1, void* input2, int32_t op) {
    if (!network || !input1 || !input2) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor1 = static_cast<nvinfer1::ITensor*>(input1);
        auto* itensor2 = static_cast<nvinfer1::ITensor*>(input2);
        auto* layer = inetwork->addElementWise(*itensor1, *itensor2, static_cast<nvinfer1::ElementWiseOperation>(op));
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_shuffle(void* network, void* input) {
    if (!network || !input) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(input);
        auto* layer = inetwork->addShuffle(*itensor);
        return layer ? layer->getOutput(0) : nullptr;
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
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_softmax(void* network, void* input, uint32_t axes) {
    if (!network || !input) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(input);
        auto* layer = inetwork->addSoftMax(*itensor);
        if (layer) {
            layer->setAxes(axes);
            return layer->getOutput(0);
        }
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_scale(void* network, void* input, int32_t mode, 
                       const void* shift, const void* scale, const void* power) {
    if (!network || !input) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(input);
        
        nvinfer1::Weights shift_w{nvinfer1::DataType::kFLOAT, shift, 0};
        nvinfer1::Weights scale_w{nvinfer1::DataType::kFLOAT, scale, 0};
        nvinfer1::Weights power_w{nvinfer1::DataType::kFLOAT, power, 0};
        
        auto* layer = inetwork->addScale(
            *itensor, 
            static_cast<nvinfer1::ScaleMode>(mode),
            shift_w, scale_w, power_w
        );
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_reduce(void* network, void* input, int32_t op, uint32_t axes, bool keep_dims) {
    if (!network || !input) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(input);
        auto* layer = inetwork->addReduce(
            *itensor,
            static_cast<nvinfer1::ReduceOperation>(op),
            axes,
            keep_dims
        );
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_slice(void* network, void* input, const int32_t* start, 
                       const int32_t* size, const int32_t* stride, int32_t nb_dims) {
    if (!network || !input || !start || !size || !stride) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(input);
        
        nvinfer1::Dims start_dims, size_dims, stride_dims;
        start_dims.nbDims = size_dims.nbDims = stride_dims.nbDims = nb_dims;
        
        for (int32_t i = 0; i < nb_dims && i < nvinfer1::Dims::MAX_DIMS; ++i) {
            start_dims.d[i] = start[i];
            size_dims.d[i] = size[i];
            stride_dims.d[i] = stride[i];
        }
        
        auto* layer = inetwork->addSlice(*itensor, start_dims, size_dims, stride_dims);
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_resize(void* network, void* input) {
    if (!network || !input) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(input);
        auto* layer = inetwork->addResize(*itensor);
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_topk(void* network, void* input, int32_t op, int32_t k, uint32_t axes) {
    if (!network || !input) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* itensor = static_cast<nvinfer1::ITensor*>(input);
        auto* layer = inetwork->addTopK(
            *itensor,
            static_cast<nvinfer1::TopKOperation>(op),
            k,
            axes
        );
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_gather(void* network, void* data, void* indices, int32_t axis) {
    if (!network || !data || !indices) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* data_tensor = static_cast<nvinfer1::ITensor*>(data);
        auto* indices_tensor = static_cast<nvinfer1::ITensor*>(indices);
        auto* layer = inetwork->addGather(*data_tensor, *indices_tensor, axis);
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_select(void* network, void* condition, void* then_input, void* else_input) {
    if (!network || !condition || !then_input || !else_input) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* condition_tensor = static_cast<nvinfer1::ITensor*>(condition);
        auto* then_tensor = static_cast<nvinfer1::ITensor*>(then_input);
        auto* else_tensor = static_cast<nvinfer1::ITensor*>(else_input);
        auto* layer = inetwork->addSelect(*condition_tensor, *then_tensor, *else_tensor);
        return layer ? layer->getOutput(0) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_assertion(void* network, void* condition, const char* message) {
    if (!network || !condition) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* condition_tensor = static_cast<nvinfer1::ITensor*>(condition);
        auto* layer = inetwork->addAssertion(*condition_tensor, message ? message : "");
        // Assertion layers don't have outputs, return the layer itself
        return layer;
    } catch (...) {
        return nullptr;
    }
}

void* network_add_loop(void* network) {
    if (!network) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        return inetwork->addLoop();
    } catch (...) {
        return nullptr;
    }
}

void* network_add_if_conditional(void* network) {
    if (!network) return nullptr;
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        return inetwork->addIfConditional();
    } catch (...) {
        return nullptr;
    }
}

// Tensor methods
const char* tensor_get_name(void* tensor) {
    if (!tensor) return nullptr;
    try {
        auto* itensor = static_cast<nvinfer1::ITensor*>(tensor);
        return itensor->getName();
    } catch (...) {
        return nullptr;
    }
}

void tensor_set_name(void* tensor, const char* name) {
    if (!tensor || !name) return;
    try {
        auto* itensor = static_cast<nvinfer1::ITensor*>(tensor);
        itensor->setName(name);
    } catch (...) {
        // Ignore errors
    }
}

void* tensor_get_dimensions(void* tensor, int32_t* dims, int32_t* nb_dims) {
    if (!tensor || !dims || !nb_dims) return nullptr;
    try {
        auto* itensor = static_cast<nvinfer1::ITensor*>(tensor);
        nvinfer1::Dims dimensions = itensor->getDimensions();
        *nb_dims = dimensions.nbDims;
        for (int32_t i = 0; i < dimensions.nbDims && i < nvinfer1::Dims::MAX_DIMS; ++i) {
            dims[i] = dimensions.d[i];
        }
        return tensor; // Return success
    } catch (...) {
        return nullptr;
    }
}

int32_t tensor_get_type(void* tensor) {
    if (!tensor) return -1;
    try {
        auto* itensor = static_cast<nvinfer1::ITensor*>(tensor);
        return static_cast<int32_t>(itensor->getType());
    } catch (...) {
        return -1;
    }
}

void* builder_build_serialized_network(void* builder, void* network, void* config, size_t* out_size) {
    if (!builder || !network || !config || !out_size) return nullptr;
    try {
        auto* ibuilder = static_cast<nvinfer1::IBuilder*>(builder);
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* iconfig = static_cast<nvinfer1::IBuilderConfig*>(config);
        
        auto* serialized = ibuilder->buildSerializedNetwork(*inetwork, *iconfig);
        if (!serialized) return nullptr;
        
        *out_size = serialized->size();
        // Allocate and copy data
        void* data = malloc(*out_size);
        if (data) {
            memcpy(data, serialized->data(), *out_size);
        }
        delete serialized;
        return data;
    } catch (...) {
        return nullptr;
    }
}

// Runtime methods
void* runtime_deserialize_cuda_engine(void* runtime, const void* data, size_t size) {
    if (!runtime || !data) return nullptr;
    try {
        auto* iruntime = static_cast<nvinfer1::IRuntime*>(runtime);
        return iruntime->deserializeCudaEngine(data, size);
    } catch (...) {
        return nullptr;
    }
}

// Engine methods
int32_t engine_get_nb_io_tensors(void* engine) {
    if (!engine) return 0;
    try {
        auto* iengine = static_cast<nvinfer1::ICudaEngine*>(engine);
        return iengine->getNbIOTensors();
    } catch (...) {
        return 0;
    }
}

const char* engine_get_tensor_name(void* engine, int32_t index) {
    if (!engine) return nullptr;
    try {
        auto* iengine = static_cast<nvinfer1::ICudaEngine*>(engine);
        return iengine->getIOTensorName(index);
    } catch (...) {
        return nullptr;
    }
}

void* engine_create_execution_context(void* engine) {
    if (!engine) return nullptr;
    try {
        auto* iengine = static_cast<nvinfer1::ICudaEngine*>(engine);
        return iengine->createExecutionContext();
    } catch (...) {
        return nullptr;
    }
}

// ExecutionContext methods
bool context_set_tensor_address(void* context, const char* name, void* data) {
    if (!context || !name) return false;
    try {
        auto* icontex = static_cast<nvinfer1::IExecutionContext*>(context);
        return icontex->setTensorAddress(name, data);
    } catch (...) {
        return false;
    }
}

bool context_enqueue_v3(void* context, void* stream) {
    if (!context) return false;
    try {
        auto* icontext = static_cast<nvinfer1::IExecutionContext*>(context);
        return icontext->enqueueV3(static_cast<cudaStream_t>(stream));
    } catch (...) {
        return false;
    }
}

// Parser methods
bool parser_parse(void* parser, const void* data, size_t size) {
    if (!parser || !data) return false;
    try {
        auto* iparser = static_cast<nvonnxparser::IParser*>(parser);
        return iparser->parse(data, size);
    } catch (...) {
        return false;
    }
}

int32_t parser_get_nb_errors(void* parser) {
    if (!parser) return 0;
    try {
        auto* iparser = static_cast<nvonnxparser::IParser*>(parser);
        return iparser->getNbErrors();
    } catch (...) {
        return 0;
    }
}

void* parser_get_error(void* parser, int32_t index) {
    if (!parser) return nullptr;
    try {
        auto* iparser = static_cast<nvonnxparser::IParser*>(parser);
        return const_cast<nvonnxparser::IParserError*>(iparser->getError(index));
    } catch (...) {
        return nullptr;
    }
}

const char* parser_error_desc(void* error) {
    if (!error) return nullptr;
    try {
        auto* ierror = static_cast<nvonnxparser::IParserError*>(error);
        return ierror->desc();
    } catch (...) {
        return nullptr;
    }
}

// CUDA wrappers
int32_t cuda_malloc_wrapper(void** ptr, size_t size) {
    return static_cast<int32_t>(cudaMalloc(ptr, size));
}

int32_t cuda_free_wrapper(void* ptr) {
    return static_cast<int32_t>(cudaFree(ptr));
}

int32_t cuda_memcpy_wrapper(void* dst, const void* src, size_t count, int32_t kind) {
    return static_cast<int32_t>(cudaMemcpy(dst, src, count, static_cast<cudaMemcpyKind>(kind)));
}

int32_t cuda_device_synchronize_wrapper() {
    return static_cast<int32_t>(cudaDeviceSynchronize());
}

const char* cuda_get_error_string_wrapper(int32_t error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}

// Destruction methods
void delete_builder(void* builder) {
    if (builder) {
        delete static_cast<nvinfer1::IBuilder*>(builder);
    }
}

void delete_network(void* network) {
    if (network) {
        delete static_cast<nvinfer1::INetworkDefinition*>(network);
    }
}

void delete_config(void* config) {
    if (config) {
        delete static_cast<nvinfer1::IBuilderConfig*>(config);
    }
}

void delete_runtime(void* runtime) {
    if (runtime) {
        delete static_cast<nvinfer1::IRuntime*>(runtime);
    }
}

void delete_engine(void* engine) {
    if (engine) {
        delete static_cast<nvinfer1::ICudaEngine*>(engine);
    }
}

void delete_context(void* context) {
    if (context) {
        delete static_cast<nvinfer1::IExecutionContext*>(context);
    }
}

void delete_parser(void* parser) {
    if (parser) {
        delete static_cast<nvonnxparser::IParser*>(parser);
    }
}

} // extern "C"
