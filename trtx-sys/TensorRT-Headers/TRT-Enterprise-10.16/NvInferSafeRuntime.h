/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// @brief Main header file for the NVIDIA Safe Runtime API.
// This file provides the primary interface for users to interact with the NVIDIA Safe Runtime API.
// It includes the necessary definitions, classes, and functions for creating and managing safe graphs,
// executing inference, and handling errors.
// Users should include this header file in their application to access the Safe Runtime API functionality.

#ifndef NV_INFER_SAFE_RUNTIME_H
#define NV_INFER_SAFE_RUNTIME_H
#include "NvInferSafeMemAllocator.h"
#include "NvInferSafePlugin.h"
#include "NvInferSafeRecorder.h"
#include <algorithm>
#include <cuda_fp16.h>

namespace nvinfer2
{
namespace safe
{
using half_t = __half;

//!
//! \class TypedArray
//!
//! \brief A standard_layout and trivially_copyable typed array that knows the
//! data type and size it is holding
//!
//! \warning The user of this class is responsible for making sure the data
//! pointer is set to a valid piece of memory
//!
//! \pre user provides a valid data ptr for this object before passing it to
//! TRT's API
//!
class TypedArray
{
public:
    TypedArray() noexcept
        : mType(DataType::kFLOAT)
        , mData(nullptr)
        , mBufferSize(0U)
    {
    }
    // Shallow copy of TypedArray is allowed
    TypedArray(TypedArray const&) = default;
    TypedArray(TypedArray&&) = default;
    TypedArray& operator=(TypedArray const&) & = default;
    TypedArray& operator=(TypedArray&&) & = default;
    ~TypedArray() noexcept = default;

    //! sets the data to a float ptr. It also sets the current type to kFLOAT
    TypedArray(float* ptr, uint64_t const bufferSize) noexcept
        : mType(DataType::kFLOAT)
        , mData(ptr)
        , mBufferSize(bufferSize)
    {
    }

    //! sets the data to a half_t ptr. It also sets the current type to kHALF
    TypedArray(half_t* ptr, uint64_t const bufferSize) noexcept
        : mType(DataType::kHALF)
        , mData(ptr)
        , mBufferSize(bufferSize)
    {
    }

    //! sets the data to a int64_t ptr. It also sets the current type to kINT64
    TypedArray(int64_t* ptr, uint64_t const bufferSize) noexcept
        : mType(DataType::kINT64)
        , mData(ptr)
        , mBufferSize(bufferSize)
    {
    }

    //! sets the data to a int32_t ptr. It also sets the current type to kINT32
    TypedArray(int32_t* ptr, uint64_t const bufferSize) noexcept
        : mType(DataType::kINT32)
        , mData(ptr)
        , mBufferSize(bufferSize)
    {
    }

    //! sets the data to a int8_t ptr. It also sets the current type to kINT8
    TypedArray(int8_t* ptr, uint64_t const bufferSize) noexcept
        : mType(DataType::kINT8)
        , mData(ptr)
        , mBufferSize(bufferSize)
    {
    }

    //! sets the data to a uint8_t ptr. It also sets the current type to kUINT8
    TypedArray(uint8_t* ptr, uint64_t const bufferSize) noexcept
        : mType(DataType::kUINT8)
        , mData(ptr)
        , mBufferSize(bufferSize)
    {
    }
    //! sets the data to a bool ptr. It also sets the current type to kBOOL
    TypedArray(bool* ptr, uint64_t const bufferSize) noexcept
        : mType(DataType::kBOOL)
        , mData(ptr)
        , mBufferSize(bufferSize)
    {
    }

    // NOLINTBEGIN to avoid clang-tidy requiring [[nodiscard]]
    //! \brief This method returns the current type of the data.
    DataType getType() const noexcept
    {
        return mType;
    }

    //! \brief Retrieves the data as float*. It should only be called when the current
    //! type is kFLOAT. Returns nullptr otherwise.
    float* getFloat() const noexcept
    {
        if (mType == DataType::kFLOAT)
        {
            return static_cast<float*>(mData);
        }
        return nullptr;
    }

    //! \brief Retrieves the data as half_t*. It should only be called when the current
    //! type is kFLOAT. Returns nullptr otherwise.
    half_t* getHalf() const noexcept
    {
        if (mType == DataType::kHALF)
        {
            return static_cast<half_t*>(mData);
        }
        return nullptr;
    }

    //! \brief Retrieves the data as int64_t*. It should only be called when the current
    //! type is kINT64. Returns nullptr otherwise.
    int64_t* getInt64() const noexcept
    {
        if (mType == DataType::kINT64)
        {
            return static_cast<int64_t*>(mData);
        }
        return nullptr;
    }

    //! \brief Retrieves the data as int32_t*. It should only be called when the current
    //! type is kINT32. Returns nullptr otherwise.
    int32_t* getInt32() const noexcept
    {
        if (mType == DataType::kINT32)
        {
            return static_cast<int32_t*>(mData);
        }
        return nullptr;
    }

    //! \brief Retrieves the data as int8_t*. It should only be called when the current
    //! type is kINT8. Returns nullptr otherwise.
    int8_t* getInt8() const noexcept
    {
        if (mType == DataType::kINT8)
        {
            return static_cast<int8_t*>(mData);
        }
        return nullptr;
    }

    //! \brief Retrieves the data as uint8_t*. It should only be called when the current
    //! type is kUINT8. Returns nullptr otherwise.
    uint8_t* getUint8() const noexcept
    {
        if (mType == DataType::kUINT8)
        {
            return static_cast<uint8_t*>(mData);
        }
        return nullptr;
    }

    //! \brief Retrieves the data as bool*. It should only be called when the current
    //! type is kBOOL. Returns nullptr otherwise.
    bool* getBool() const noexcept
    {
        if (mType == DataType::kBOOL)
        {
            return static_cast<bool*>(mData);
        }
        return nullptr;
    }

    //! \brief Retrieves the size of the array in bytes.
    uint64_t getSize() const noexcept
    {
        return mBufferSize;
    }

    //! \brief Retrieves the data regardless of the type
    //! \note This is an internal method, not to be used outside of TensorRT safe runtime APIs.
    void* getData() const noexcept
    {
        return mData;
    }
    // NOLINTEND

private:
    DataType mType;       // This is the current type of the data.
    void* mData;          // This is the pointer that holds the data.
    uint64_t mBufferSize; // This is the size of the buffer in bytes that holds the data.
};

//!
//! \class PhysicalDims
//! \brief Structure to define the physical dimensions of a tensor with support for up to 9 dimensions.
//!
//! This structure is specifically designed for describing physical tensors layout in memory
//! which can have up to 9 dimensions after vectorization.
//! Unlike the standard Dims structure which is limited to 8 dimensions,
//! PhysicalDims supports the additional dimension needed when tensors are vectorized.
//!
//! When a tensor is vectorized by size T along some dimension with size K, the vectorized
//! dimension is split into two dimensions of sizes ceil(K/T) and T, potentially resulting
//! in 9 physical dimensions for an originally 8 logical dimensional tensor.
//!
class PhysicalDims
{
public:
    //! The maximum rank (number of dimensions) supported for physical tensor layout.
    //! This is set to 9 to accommodate vectorized tensors that split one dimension into two.
    static constexpr int32_t MAX_DIMS{9};

    //! The rank (number of dimensions).
    int32_t nbDims;

    //! The extent of each dimension.
    int64_t d[MAX_DIMS];
};

inline bool operator==(PhysicalDims const& d0, PhysicalDims const& d1) noexcept
{
    return d0.nbDims == d1.nbDims && (d0.nbDims <= 0 || std::equal(d0.d, d0.d + d0.nbDims, d1.d));
}

inline bool operator!=(PhysicalDims const& d0, PhysicalDims const& d1) noexcept
{
    return !(d0 == d1);
}

//!
//! \struct TensorDescriptor
//!
//! \brief A simple record summarizing various properties of a network IO Tensor.
//!
//! All properties needed to query, allocate, read, and write IO Tensors are recorded
//! here. The user can determine the physical tensor shape and layout used by the compiler,
//! which may be different from the original user-specified shape due to padding or tiling
//! for vectorized layouts. For example, a user-specified shape [1,10,3,4] with a TensorFormat
//! specification of NHWC8 would lead to tiling by 8 elements on the C dimension with the
//! following descriptor fields:
//!
//! User-specified properties (NHWC8):
//! userShape             : [1,10,3,4] (in NCHW order)
//! componentsPerVector   : 8
//! vectorizedDim         : 1 (C)
//!
//! Compiler-determined layout:
//! shape                 : [1,2,8,3,4] (C dimension is split into two dims)
//! stride                : [192,8,1,64,16]
//! strideOrder           : [4,1,0,3,2] (Tiled C dimensions are changing fastest)
//!
struct TensorDescriptor
{
    //! Name of the IO Tensor.
    AsciiChar const* tensorName{nullptr};
    //! Static shape of the IO Tensor.
    //! \note The static shape will depend on the IO Profile selected.
    PhysicalDims shape{-1, {}};
    //! Stride vector for each element of the IO Tensor.
    PhysicalDims stride{-1, {}};
    //! DataType enum for the IO Tensor.
    //! \warning The actual type of the tensor data must correspond to this DataType.
    DataType dataType{DataType::kFLOAT};
    //! The size of the tensor data type in bytes (4 for float and int32, 2 for half, 1 for int8)
    uint64_t bytesPerComponent{0U};
    //! The vector length (in scalars) for a vectorized tensor, 1 if the tensor is not vectorized.
    uint64_t componentsPerVector{1U};
    //! The dimension index along which the tensor is vectorized, -1 if the tensor is not vectorized.
    int64_t vectorizedDim{-1};
    //! Total size in bytes for the IO Tensor.
    uint64_t sizeInBytes{0U};
    //! Enum to denote whether the Tensor is for input or output.
    TensorIOMode ioMode{TensorIOMode::kNONE};
    //! Enum to denote whether the Tensor memory is allocated on the GPU, CPU, or CPU_PINNED.
    MemoryPlacement memPlacement{MemoryPlacement::kNONE};
    //! The order in which the dimensions are laid out in memory.
    PhysicalDims strideOrder{-1, {}};
    //! The original user-specified shape of the tensor, provided as a reference if the tensor is
    //! vectorized. When a tensor is vectorized by size T along some dimension with size K, the
    //! vectorized dimension is split into two dimensions of sizes ceil(K/T) and T.
    Dims userShape{-1, {}};
};

//!
//! \class ITRTGraph
//!
//! \brief Abstract Interface for a functionally safe graph for executing inference on a built network.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \warning APIs are not thread safe: Users must guarantee that each concurrently running thread uses a unique instance
//! (clone) of the API object.
//!
//! \note The set APIs are not thread safe and get APIs are thread-safe.
//!
class ITRTGraph
{
public:
    ITRTGraph(ITRTGraph const&) = delete;
    ITRTGraph(ITRTGraph&&) = delete;
    ITRTGraph& operator=(ITRTGraph const&) & = delete;
    ITRTGraph& operator=(ITRTGraph&&) & = delete;

    //! \brief A shallow destructor of ITRTGraph
    //!
    //! \pre DEINIT API
    virtual ~ITRTGraph() noexcept = default;

    //! \brief Specialized Graph shallow copy
    //!
    //! \details This function constructs a new ITRTGraph which uses a shared pointer to the persistent
    //! part of the graph (pointing to the same set of weights) so we do not duplicate the weights anywhere. Clone will
    //! automatically allocate a new scratch memory if the scratch memory is trtManaged (trtManagedScratch set to true
    //! in createTRTGraph) otherwise user has to call setScratchMemory before calling execute.  The cloned graph
    //! will be using the same memory allocator as the original graph.
    //!
    //! \pre INIT API
    //!
    //! \param graph A reference to an ITRTGraph pointer, that will be initialized after this call.
    //! \param recorder An ISafeRecorder that records the errors happening during graph initialization and inference.
    //!
    //! \note Each cloned graph may be given it's own recorder so there is an independent source of truth for each
    //! graph.
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode clone(ITRTGraph*& graph, ISafeRecorder& recorder) noexcept = 0;

    //! \brief This function returns the scratch memory size (in bytes) needed to store all the
    //! intermediate tensors for inference. The user could allocate a scratch memory of this size and pass it
    //! to TRT via setScratchMemory if not using trtManagedScratch.
    //!
    //! \pre INIT API
    //!
    //! \see setScratchMemory
    //!
    //! \param size The return value for the scratch memory size (in bytes) needed to store all the
    //!        intermediate tensors for inference.
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getScratchMemorySize(size_t& size) const noexcept = 0;

    //! \brief This function returns the trtManagedScratch flag provided in createTRTGraph call.
    //!
    //! \pre INIT API
    //!
    //! \see createTRTGraph
    //!
    //! \param flag A boolean flag that determines if scratch memory is managed by TRT
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getTRTManagedScratch(bool& flag) const noexcept = 0;

    //! \brief This function sets the scratch memory for the graph. This should only be called if scratch memory
    //! is not TRT managed (trtManagedScratch is false). An error will be recorded if called on a graph where
    //! trtManagedScratch is true.
    //!
    //! \pre INIT API
    //!
    //! \param memory pointer to a device memory block allocated by user that is at least as large as
    //!        getScratchMemorySize, or nullptr which will reset the internal pointer.
    //!
    //! \see getScratchMemorySize
    //!
    //! \return ErrorCode
    //!
    //! \warning User must guarantee that the allocated scratch memory is large enough.
    //!
    virtual ErrorCode setScratchMemory(void* memory) noexcept = 0;

    //! \brief This function gets the scratch memory for the graph. This should only be called if scratch memory
    //! is not TRT managed (trtManagedScratch is false). An error will be recorded if called on a graph where
    //! trtManagedScratch is true.
    //!
    //! \pre INIT API
    //!
    //! \param memory set to a device memory pointer allocated by user or nullptr when the scratch memory is TRT
    //! managed.
    //!
    //! \see getScratchMemorySize
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getScratchMemory(void*& memory) noexcept = 0;

    //! \brief This function returns the total number of input and output tensor for the current graph.
    //!
    //! \pre INIT API
    //!
    //! \param nb The return value for total number of input and output tensor for the current graph.
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getNbIOTensors(int64_t& nb) const noexcept = 0;

    //! \brief This function returns the name of a tensor for a given index.
    //!
    //! \pre INIT API
    //!
    //! \param name The name string of the tensor that will be filled out after the call.
    //! \param index The index of the tensor whose name is to be returned.
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getIOTensorName(AsciiChar const*& name, size_t const index) const noexcept = 0;

    //! \brief This function should return a TensorDescriptor which contains all the information about the tensor based
    //! on the name.
    //!
    //! \pre INIT API
    //!
    //! \param desc corresponding TensorDescriptor that will be filled out after the call.
    //! \param name name of the tensor we are interested in.
    //!
    //! \return ErrorCode
    //! \warning If the name does not correspond to a valid IO tensor, the function will fail with an error code
    //! of ErrorCode::kINVALID_ARGUMENT
    //!
    virtual ErrorCode getIOTensorDescriptor(
        TensorDescriptor& desc, AsciiChar const* const tensorName) const noexcept = 0;

    //! \brief This function should return a TensorDescriptor which contains all the information about the tensor based
    //! on the index.
    //!
    //! \pre INIT API
    //!
    //! \param desc corresponding TensorDescriptor that will be filled out after the call.
    //! \param index index of the tensor we are interested in (starting from 0).
    //!
    //! \return ErrorCode
    //! \warning If the index does not fall between 0 and getNbIOTensors()-1, the function will fail with an error code
    //! of ErrorCode::kINVALID_ARGUMENT
    //!
    virtual ErrorCode getIOTensorDescriptor(TensorDescriptor& desc, int32_t const index) const noexcept = 0;

    //! \brief This function assigns a user allocated device memory block for an input tensor to the graph based on its
    //! name.
    //!
    //! \pre INIT API
    //!
    //! \param tensorName the tensor name that the user would like to provide memory for.
    //! \param tensor a device memory block allocated by user for the tensor.
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode setIOTensorAddress(AsciiChar const* const tensorName, TypedArray const& tensor) noexcept = 0;

    //! \brief This function assigns a user allocated device memory block for an input tensor to the graph based on its
    //! index.
    //!
    //! \pre INIT API
    //!
    //! \param index the tensor index that the user would like to provide memory for.
    //! \param tensor a device memory block allocated by user for the tensor.
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode setIOTensorAddress(int32_t const index, TypedArray const& tensor) noexcept = 0;

    //! \brief This function gets the memory address for an user provided input tensor to the graph based on its
    //! name.
    //!
    //! \pre INIT API
    //!
    //! \param tensorName the tensor name that the user would like to obtain memory for.
    //! \param tensor set to a device memory pointer allocated by user for the tensor.
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getIOTensorAddress(AsciiChar const* const tensorName, TypedArray& tensor) noexcept = 0;

    //! \brief This function gets the memory address for an user provided input tensor to the graph based on its
    //! index.
    //!
    //! \pre INIT API
    //!
    //! \param index the tensor index that the user would like to obtain memory for.
    //! \param tensor set to a device memory pointer allocated by user for the tensor.
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getIOTensorAddress(int32_t const index, TypedArray& tensor) noexcept = 0;

    //! \brief This function sets a cudaEvent on the current graph that triggers when the input is consumed.
    //! At that point, the input memory can be recycled, i.e. new input for next inference can be loaded.
    //!
    //! \pre INIT API
    //!
    //! \param event cuda event to be set
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode setInputConsumedEvent(cudaEvent_t event) noexcept = 0;

    //! \brief This function retrieves the cudaEvent on the current graph that triggers when the input is
    //! fully consumed. At that point, the input memory can be recycled, i.e. new input for next inference
    //! can be loaded.
    //!
    //! \pre RUNTIME API
    //!
    //! \param event retrieved cuda event
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getInputConsumedEvent(cudaEvent_t& event) const noexcept = 0;

    //! \brief This function retrieves the RuntimeErrorInformation (for async error) buffer for the current graph.
    //! This buffer includes all the runtime error types such as gather out of bound, silently consumed NaN value etc.
    //!
    //! \pre RUNTIME API
    //!
    //! \param buffer retrieved error buffer.
    //!
    //! \see RuntimeErrorInformation
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getErrorBuffer(RuntimeErrorInformation*& buffer) const noexcept = 0;

    //! \brief This function retrieves the ISafeRecorder for the current graph.
    //!
    //! \pre RUNTIME API
    //!
    //! \param recorder retrieved ISafeRecorder.
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getSafeRecorder(ISafeRecorder*& recorder) const noexcept = 0;

    //!
    //! \brief This function returns the total number of IO tensor profiles for the current graph.
    //!
    //! \pre RUNTIME API
    //!
    //! \param nb The return value for the total number of IO tensor profiles for the current graph.
    //!           A graph will at least have 1 IOProfile
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getNbIOProfiles(int64_t& nb) const noexcept = 0;

    //!
    //! \brief This function selects the active IOProfile for the graph. If this function is not called,
    //! the TRTGraph will default to profile 0. Each IOProfile on the graph is mutually exclusive, meaning
    //! only one IOProfile can be active at a time.
    //!
    //! \pre RUNTIME API
    //!
    //! \param profileIndex The index of the profile to select.
    //!
    //! \return ErrorCode
    //!
    //! \note Other graph APIs are executed in the context of the current IOProfile id, so IOProfile id should be
    //! changed only on synchronized inference boundaries.
    //!
    virtual ErrorCode setIOProfile(int64_t profileIndex) noexcept = 0;

    //!
    //! \brief This function retrieves the index of the current active IOProfile for the graph.
    //!
    //! \pre RUNTIME API
    //!
    //! \param profileIndex retrieved profile index.
    //!
    //! \return ErrorCode
    //!
    virtual ErrorCode getIOProfile(int64_t& profileIndex) const noexcept = 0;

    //! \brief Return the number of auxiliary streams used by this graph.
    //!
    //! \pre INIT API
    //!
    //! \param nbStreams The return value for the number of auxiliary streams.
    //!
    //! \return ErrorCode
    //!
    //! \see setAuxStreams()
    //!
    virtual ErrorCode getNbAuxStreams(int32_t& nbStreams) const noexcept = 0;

    //! \brief Set the auxiliary streams that TensorRT should use to run kernels on.
    //!
    //! \details TRT will launch the kernels that are supposed to run on the auxiliary streams
    //! using the streams provided by the user via this API. The user is responsible for
    //! allocating and deallocating these streams.
    //!
    //! If getNbAuxStreams() returns a value greater than 0, this API must be called before
    //! executeAsync() to provide the required auxiliary streams.
    //!
    //! If getNbAuxStreams() returns 0, setAuxStreams() can only be called with an array of size 0.
    //!
    //! \pre INIT API
    //!
    //! \param auxStreams The pointer to an array of cudaStream_t with the array length equal to nbStreams.
    //!        All streams in the array must be valid CUDA streams.
    //! \param nbStreams The number of auxiliary streams provided. Must be equal to the value returned by
    //!        getNbAuxStreams(). If nbStreams does not match, kINVALID_ARGUMENT will be returned.
    //!
    //! \return ErrorCode kSUCCESS on success, kINVALID_ARGUMENT if auxStreams contains nullptr,
    //!         or if nbStreams does not match getNbAuxStreams().
    //!
    //! \see getNbAuxStreams()
    //!
    virtual ErrorCode setAuxStreams(cudaStream_t* auxStreams, int32_t nbStreams) noexcept = 0;

    //! \brief execute one inference of this graph.
    //!
    //! \pre RUNTIME API
    //!
    //! \param stream A CUDA main stream on which the inference kernels will be enqueued. Must be a valid CUDA stream.
    //!
    //! \return ErrorCode kSUCCESS on success, if any execution error occurred other error code might be returned
    //! Errors may include but not be limited to:
    //! - Internal errors during executing one engine layer (host side)
    //! - CUDA errors
    //! - Some input or output tensor addresses have not been set.
    //!
    virtual ErrorCode executeAsync(cudaStream_t stream) noexcept = 0;

    //! \brief synchronize one inference of this graph.
    //!
    //! \pre RUNTIME API
    //!
    //! \return ErrorCode kSUCCESS on success, if any execution error occurred other error code might be returned.
    //! Errors may include but not be limited to:
    //! - Internal errors during executing one engine layer (host side)
    //! - CUDA errors
    //! - Some input or output tensor addresses have not been set.
    //!
    virtual ErrorCode sync() noexcept = 0;

protected:
    ITRTGraph() = default;
};

//! \brief The C factory function which serves as an entry point to TRT that creates an instance of a ITRTGraph class.
//! It will performs the following operations:
//! - Deserialize the TRT engine (Serialized binary blob) from the buffer
//! - Allocate the persistent device memory (for const model weights) though the provided allocator
//!   (if not provided then using the default allocator), load the weights into device memory and store
//!   it as a shared_ptr member variable (for sharing with clone method)
//!
//! \pre INIT API
//!
//! \param graph A pointer to ITRTGraph, that will be initialized after this call.
//! \param buffer A buffer that holds the serialized blob that describes the graph.
//! \param bufferSize Size of the serialized blob
//! \param recorder An ISafeRecorder that sets severity level and records the error messages during the inference. Note
//!        this does not include RuntimeError happening during kernel execution, see getErrorBuffer for RuntimeError.
//! \param trtManagedScratch if this parameter is true (default), then TRT will use the memory allocator to allocate
//!        the scratch memory that holds the intermediate tensors. When cloning the graph, TRT will also
//!        allocate a new scratch memory automatically. If the parameter is set to false (for sharing scratch memory
//!        across multiple graphs), user needs to call setScratchMemory for the graph (and its clones) with a memory
//!        buffer size at least the size queried from getScratchMemorySize.
//! \param allocator The memory allocator to be used by the runtime. All GPU memory acquired will use
//!        this allocator. If nullptr is passed, the default allocator will be used, which calls cudaMalloc and
//!        cudaFree.
//!
//! \return ErrorCode
extern "C" ErrorCode createTRTGraph(ITRTGraph*& graph, void const* buffer, int64_t bufferSize, ISafeRecorder& recorder,
    bool trtManagedScratch = true, ISafeMemAllocator* allocator = nullptr) noexcept;

//!
//! \brief Toplevel graph destructor.
//! - Frees the shared persistent memory only if this is the last graph instance that holds a reference to it.
//! - Frees the scratch memory if it is managed by TRT
//! - Deletes the graph object and all instruction instances underneath it for all IO profiles
//!
//! \pre DEINIT API
//!
//! \param graph A pointer reference to a graph object which will be destroyed and reference will be set to nullptr.
//! \return ErrorCode
extern "C" ErrorCode destroyTRTGraph(ITRTGraph*& graph) noexcept;

} // namespace safe
} // namespace nvinfer2
#endif /* NV_INFER_SAFE_RUNTIME_H */
