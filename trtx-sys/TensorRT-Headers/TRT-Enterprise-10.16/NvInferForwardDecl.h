/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
using Dims = nvinfer1::Dims;
using IVersionedInterface = nvinfer1::IVersionedInterface;
using AsciiChar = nvinfer1::AsciiChar;
using InterfaceInfo = nvinfer1::InterfaceInfo;
using DataType = nvinfer1::DataType;
using TensorIOMode = nvinfer1::TensorIOMode;
using ErrorCode = nvinfer1::ErrorCode;
using IErrorRecorder = nvinfer1::IErrorRecorder;

// NvInferPluginBase.h classes
using IPluginResource = nvinfer1::IPluginResource;
using IPluginCreatorInterface = nvinfer1::IPluginCreatorInterface;
using IPluginV3 = nvinfer1::IPluginV3;
using IPluginCapability = nvinfer1::IPluginCapability;
using PluginField = nvinfer1::PluginField;
using PluginFieldCollection = nvinfer1::PluginFieldCollection;
using TensorRTPhase = nvinfer1::TensorRTPhase;

// Re implement EnumMax in the nvinfer2::safe namespace
namespace impl
{
//! Declaration of EnumMaxImpl struct to store maximum number of elements in an enumeration type.
template <typename T>
struct EnumMaxImpl;
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
