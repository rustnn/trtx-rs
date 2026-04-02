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

// Header file for the NVIDIA Safe Runtime Memory Allocator interface.
// This file provides the user-implementable interface for the Memory Allocator feature in the NVIDIA Safe Runtime API.
// It includes the necessary classes, functions, and definitions for creating and managing safe memory allocators,
// including the ISafeMemAllocator interface.
// Users should implement this interface to provide a custom memory allocator that will be used by the Safe Runtime
// to allocate memory for both CPU and GPU resources. The allocator should handle memory allocation and deallocation
// requests from the Safe Runtime, and ensure thread safety and proper memory management.
// The implemented allocator will be responsible for allocating memory for various components of the Safe Runtime,
// including tensors, plugins, and other internal data structures.

#ifndef NV_INFER_SAFE_MEM_ALLOCATOR_H
#define NV_INFER_SAFE_MEM_ALLOCATOR_H
#include "NvInferSafeRecorder.h"
#include <cstdint>

namespace nvinfer2
{
namespace safe
{
class ISafeRecorder;

//! \brief Legacy constant (16 GiB) kept for TensorRT excessiveMemoryDetection test build compatibility.
//! \remark TRTS-11093 removed the max alloc limit from the allocator; this constant is deprecated and
//!         will be removed once TensorRT updates its lweSafety tests. Do not use for new code.
constexpr uint64_t kMAXIMUM_ALLOC_SIZE{17179869184U};

//!
//! \enum MemoryPlacement
//! \brief Enum to describe the placement of the memory region
//!
enum class MemoryPlacement : uint32_t
{
    //! Invalid or unspecified placement (used for error checking)
    kNONE = 0x0,
    //! CUDA managed memory (not used for safety)
    kMANAGED = 0x1,
    //! Device memory allocated via cudaMalloc
    kGPU = 0x2,
    //! Regular (paged) host memory allocated via malloc, or aligned_alloc
    kCPU = 0x4,
    //! Page-locked host memory allocated via cudaHostAlloc, mappable to device (for zero-copy)
    kCPU_PINNED = 0x80
};

//!
//! \enum MemoryUsage
//! \brief Enum to describe the usage of memory region
//!
enum class MemoryUsage : uint32_t
{
    //! Uncommitted memory usage
    kGENERIC = 0x0,
    //! Memory (network weights) that is initialized once and can be shared across different graphs
    kIMMUTABLE = 0x3,
    //! Scratchpad memory per context used for intermediate tensors
    kSCRATCH = 0x5,
    //! Memory used for IO Tensors
    kIOTENSOR = 0x6
};

//!
//! \class ISafeMemAllocator
//!
//! \brief Application-implemented class for controlling memory allocation on the GPU/CPU.
//!
//! \warning The lifetime of an ISafeMemAllocator object must exceed that of all objects that use it.
//!
class ISafeMemAllocator
{
public:
    //!
    //! \brief A thread-safe callback implemented by the application to handle acquisition of GPU/CPU memory.
    //!
    //! \pre INIT API
    //!
    //! \param size The size of the memory block required (in bytes).
    //! \param alignment The required alignment of memory. Alignment will be zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags The type of memory to be allocated:
    //!        one of MemoryPlacement::kGPU, MemoryPlacement::kCPU, MemoryPlacement::kCPU_PINNED.
    //! \param usage The intended usage of memory to be allocated:
    //!        one of MemoryUsage::kIMMUTABLE, MemoryUsage::kSCRATCH, MemoryUsage::kIOTENSOR.
    //! \param recorder A safe message recorder
    //!
    //! \return If the allocation was successful, the start address of a device memory block of
    //!         the requested size, nullptr otherwise.
    //!         - If an allocation request of size 0 is made, nullptr must be returned.
    //!         - If an allocation request cannot be satisfied for any reason, nullptr must be returned.
    //!         - If a non-null address is returned, it is guaranteed to have the specified alignment.
    //!
    //! \warning The implementation must guarantee thread safety for concurrent allocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual void* allocate(uint64_t const size, uint64_t const alignment, MemoryPlacement const flags,
        MemoryUsage const usage, ISafeRecorder& recorder) noexcept = 0;

    virtual ~ISafeMemAllocator() = default;
    ISafeMemAllocator() = default;

    //!
    //! \brief A thread-safe callback implemented by the application to handle release of GPU/CPU memory.
    //!
    //! \pre INIT API
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory A memory address that was previously returned by an allocate() call of the same
    //!        allocator object.
    //! \param flags The type of memory to be deallocated:
    //!        one of MemoryPlacement::kGPU, MemoryPlacement::kCPU, MemoryPlacement::kCPU_PINNED.
    //! \param recorder A safe message recorder
    //!
    //! \return True if the acquired memory is released successfully, false otherwise.
    //!
    //! \warning The implementation must guarantee thread safety for concurrent allocate/deallocate
    //! requests.
    //!
    //! \note Calling deallocate() multiple times with the same pointer results in undefined behavior. Implementations
    //! MUST detect and reject double-free attempts.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual bool deallocate(void* const memory, MemoryPlacement const flags, ISafeRecorder& recorder) noexcept = 0;

protected:
    ISafeMemAllocator(ISafeMemAllocator const&) = default;
    ISafeMemAllocator(ISafeMemAllocator&&) = default;
    ISafeMemAllocator& operator=(ISafeMemAllocator const&) & = default;
    ISafeMemAllocator& operator=(ISafeMemAllocator&&) & = default;
};
} // namespace safe
} // namespace nvinfer2

#endif /* NV_INFER_SAFE_MEM_ALLOCATOR_H */
