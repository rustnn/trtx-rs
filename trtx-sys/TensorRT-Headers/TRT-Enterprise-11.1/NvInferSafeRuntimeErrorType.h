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

// Single source of truth for the RuntimeErrorType enum, shared between host code
// (via NvInferSafeRecorder.h) and NVRTC-compiled codegen headers (via
// src/external/codegen_headers/cuda_codegen/runtime_error_types.h).

#ifndef NV_INFER_SAFE_RUNTIME_ERROR_TYPE_H
#define NV_INFER_SAFE_RUNTIME_ERROR_TYPE_H

#include <stdint.h>

namespace nvinfer2
{
namespace safe
{

//! \enum RuntimeErrorType
//! \brief Enumerates types of runtime errors that can occur during kernel execution.
//! \details
//! - kNAN_CONSUMED error occurs when a NAN value is stored in an INT8 or FP4 quantized datatype.
//! - kINF_CONSUMED error occurs when a +/-INF value is stored in an INT8 or FP4 quantized datatype.
//! - kGATHER_OOB error occurs when a gather index tensor contains a value that is outside of the data tensor.
//! - kSCATTER_OOB error occurs when a scatter index tensor contains a value that is outside of the data tensor.
//! - kSCATTER_RACE error occurs when a scatter index tensor contains duplicate indices with reduction mode kNONE.
//! - kDIV_ZERO error occurs when a division-by-zero happens and its output is of an integer type.
enum class RuntimeErrorType : uint64_t
{
    kNAN_CONSUMED = 1ULL << 0, //!< NaN floating-point value was silently consumed
    kINF_CONSUMED = 1ULL << 1, //!< Inf floating-point value was silently consumed
    kGATHER_OOB = 1ULL << 2,   //!< Out-of-bounds access in gather operation
    kSCATTER_OOB = 1ULL << 3,  //!< Out-of-bounds access in scatter operation
    kSCATTER_RACE = 1ULL << 4, //!< Race condition in scatter operation
    kDIV_ZERO = 1ULL << 5,     //!< Division-by-zero in int division
};

} // namespace safe
} // namespace nvinfer2

#endif /* NV_INFER_SAFE_RUNTIME_ERROR_TYPE_H */
