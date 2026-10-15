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
#ifndef NV_INFER_FORWARD_DECL_H
#define NV_INFER_FORWARD_DECL_H

//! Allow direct inclusion of TRT Runtime Base header
#define NV_INFER_INTERNAL_INCLUDE 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "NvInferPluginBase.h"
#include "NvInferRuntimeBase.h"
#undef NV_INFER_INTERNAL_INCLUDE

#pragma GCC diagnostic pop

namespace nvinfer2
{
namespace safe
{
// NvInferRuntimeBase.h classes
using nvinfer1::Dims;
using nvinfer1::IVersionedInterface;
using nvinfer1::AsciiChar;
using nvinfer1::InterfaceInfo;
using nvinfer1::DataType;
using nvinfer1::TensorIOMode;
using nvinfer1::ErrorCode;
using nvinfer1::IErrorRecorder;

// NvInferPluginBase.h classes
using nvinfer1::IPluginResource;
using nvinfer1::IPluginCreatorInterface;
using nvinfer1::IPluginV3;
using nvinfer1::IPluginCapability;
using nvinfer1::PluginField;
using nvinfer1::PluginFieldCollection;
using nvinfer1::TensorRTPhase;

// Re implement EnumMax in the nvinfer2::safe namespace
namespace impl
{
//! Declaration of EnumMaxImpl struct to store maximum number of elements in an enumeration type.
template <typename T>
struct EnumMaxImpl
{
};
} // namespace impl

//! Maximum number of elements in an enumeration type.
template <typename T>
constexpr int32_t EnumMax() noexcept
{
    return impl::EnumMaxImpl<T>::kVALUE;
}

} // namespace safe
} // namespace nvinfer2

#endif /* NV_INFER_FORWARD_DECL_H */
