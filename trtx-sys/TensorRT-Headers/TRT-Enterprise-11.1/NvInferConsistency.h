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

#ifndef NV_INFER_CONSISTENCY_CHECKER_H
#define NV_INFER_CONSISTENCY_CHECKER_H
#include "NvInferForwardDecl.h"
#include "NvInferPluginBase.h"
#include "NvInferSafeRuntime.h"
#include <vector>

//!
//! \file NvInferConsistency.h
//!

namespace nvinfer2
{
namespace safe
{
// Forward declaration
class ISafeRecorder;

namespace consistency
{

//!
//! \class IConsistencyChecker
//!
//! \brief Validates a TRT engine (Serialized binary blob)
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConsistencyChecker
{
public:
    //!
    //! \brief Check that a blob that was input to createConsistencyChecker method represents a valid TRT engine.
    //!
    //! \return true if the original blob encoded an engine that belongs to valid TRT engine, false otherwise.
    //!
    virtual bool validate() noexcept = 0;
    //!
    //! \brief De-allocates any internally allocated memory.
    //!
    virtual ~IConsistencyChecker() = default;

protected:
    //!
    //! \brief Default constructor for IConsistencyChecker.
    //! \details Constructs an object in a default state. Access is protected; only derived types or the factory
    //!          createConsistencyChecker() may create instances. The object does not yet hold or validate any blob
    //!          until initialized by the factory.
    //! \pre none
    //! \post Object is in default-constructed state and is not yet bound to any engine blob.
    //!
    IConsistencyChecker() = default;
    //!
    //! \brief Deleted copy constructor to maintain non-copyability.
    //! \pre none
    //! \post none
    //!
    IConsistencyChecker(IConsistencyChecker const& other) = delete;
    //!
    //! \brief Deleted copy assignment operator (lvalue) to maintain non-copyability.
    //! \pre none
    //! \post none
    //!
    IConsistencyChecker& operator=(IConsistencyChecker const& other) & = delete;
    //!
    //! \brief Deleted move constructor to maintain non-movability.
    //! \pre none
    //! \post none
    //!
    IConsistencyChecker(IConsistencyChecker&& other) = delete;
    //!
    //! \brief Deleted move assignment operator (lvalue) to maintain non-movability.
    //! \pre none
    //! \post none
    //!
    IConsistencyChecker& operator=(IConsistencyChecker&& other) & = delete;
};

//!
//! \class IPluginChecker
//! \brief Interface for validating a plugin used in a TensorRT engine.
//! \details Used by the consistency checker when validating an engine blob that contains plugins.
//!          Implementations check that plugin inputs, outputs, and field collection are consistent
//!          and valid. There is no public constructor; only derived types or the consistency checker
//!          create instances.
//!
class IPluginChecker
{
public:
    //!
    //! \brief Validate a plugin.
    //!
    //! \param Inputs The input tensors of the plugin.
    //! \param Outputs The output tensors of the plugin.
    //! \param fc The plugin field collection.
    //!
    virtual bool validate(std::vector<TensorDescriptor> const& Inputs, std::vector<TensorDescriptor> const& Outputs,
        PluginFieldCollection* fc) noexcept = 0;

    //!
    //! \brief De-allocates any internally allocated memory.
    //!
    virtual ~IPluginChecker() = default;

protected:
    //!
    //! \brief Default constructor for IPluginChecker.
    //! \details Constructs an object in a default state. Access is protected; only derived types may create
    //!          instances. The object is not yet bound to any plugin validation context until used by the
    //!          consistency checker.
    //! \pre none
    //! \post Object is in default-constructed state and is not yet validating any plugin.
    //!
    IPluginChecker() = default;
    //!
    //! \brief Deleted copy constructor to maintain non-copyability.
    //! \pre none
    //! \post none
    //!
    IPluginChecker(IPluginChecker const& other) = delete;
    //!
    //! \brief Deleted copy assignment operator to maintain non-copyability.
    //! \pre none
    //! \post none
    //!
    IPluginChecker& operator=(IPluginChecker const& other) & = delete;
    //!
    //! \brief Deleted move constructor to maintain non-movability.
    //! \pre none
    //! \post none
    //!
    IPluginChecker(IPluginChecker&& other) = delete;
    //!
    //! \brief Deleted move assignment operator to maintain non-movability.
    //! \pre none
    //! \post none
    //!
    IPluginChecker& operator=(IPluginChecker&& other) & = delete;
};

//!
//! \brief Create an instance of an IConsistencyChecker class.
//!
//! ISafeRecorder is the error recorder class for the consistency checker.
//!
extern "C" nvinfer2::safe::ErrorCode createConsistencyChecker(IConsistencyChecker*& icc,
    nvinfer2::safe::ISafeRecorder& recorder, void const* blob, size_t size,
    std::vector<std::string> const& pluginBuildLibs = {}) noexcept;

} // namespace consistency

} // namespace safe
} // namespace nvinfer2

#endif // NV_INFER_CONSISTENCY_CHECKER_H
