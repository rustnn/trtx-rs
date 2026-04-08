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

// Header file for the NVIDIA Safe Runtime PluginV3 interface.
// This file provides the user-implementable interface for the PluginV3 feature in the NVIDIA Safe Runtime.
// It includes the necessary classes, functions, and definitions for creating and managing safe plugins,
// including the ISafePluginResourceContext, ISafePluginRegistry, IPluginV3OneSafeCore, IPluginV3OneSafeBuild,
// IPluginV3OneSafeRuntime, and ISafePluginCreatorV3One interfaces.
// Users should implement these interfaces to create custom plugins that can be used with in the Safe Runtime.

#ifndef NV_INFER_SAFE_PLUGIN_H
#define NV_INFER_SAFE_PLUGIN_H
#include "NvInferForwardDecl.h"
#include "NvInferPluginBase.h"
#include "NvInferSafeMemAllocator.h"
#include "NvInferSafeRecorder.h"

namespace nvinfer1
{
class DimsExprs;
class IExprBuilder;
} // namespace nvinfer1

namespace nvinfer2
{
namespace safe
{
// Forward declaration of TensorDescriptor and RuntimeErrorInformation for the plugin interface classes
struct RuntimeErrorInformation;
struct TensorDescriptor;

//!
//! \class ISafePluginResourceContext
//!
//! \brief Interface for plugins to access per context resources provided by TensorRT
//!
//! There is no public way to construct an ISafePluginResourceContext (i.e. User should not construct this object). It
//! appears as an argument to IPluginV3OneSafeRuntime::initResource(). Overrides of that method can use the
//! ISafePluginResourceContext object to access the ISafeMemAllocator and ISafeRecorder object asscociate with the
//! specific TRTGraph that the plugin is part of
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \see IPluginV3OneSafeRuntime::initResource()
//!
class ISafePluginResourceContext
{
public:
    //! \brief Get the GPU allocator associated with the resource context
    //!
    //! \pre INIT API
    //!
    //! \see IPluginV3OneSafeRuntime::initResource()
    //!
    virtual ISafeMemAllocator* getSafeMemAllocator() const noexcept = 0;

    //! \brief Get the error recorder associated with the resource context
    //!
    //! \pre RUNTIME API
    //!
    //! \see IPluginV3OneSafeRuntime::initResource()
    //!
    virtual ISafeRecorder* getSafeRecorder() const noexcept = 0;

    //! \brief Get the RuntimeErrorInformation associated with the resource context
    //!
    //! \pre RUNTIME API
    //!
    //! \see IPluginV3OneSafeRuntime::initResource()
    //!
    virtual RuntimeErrorInformation* getRuntimeErrorInformation() const noexcept = 0;

    virtual ~ISafePluginResourceContext() noexcept = default;

protected:
    ISafePluginResourceContext() = default;
    ISafePluginResourceContext(ISafePluginResourceContext const&) = default;
    ISafePluginResourceContext(ISafePluginResourceContext&&) = default;
    ISafePluginResourceContext& operator=(ISafePluginResourceContext const&) & = default;
    ISafePluginResourceContext& operator=(ISafePluginResourceContext&&) & = default;
};

//!
//! \class ISafePluginRegistry
//
//! \brief PluginRegistry interface class that contains all the methods user can use to interface with the registry
//! object.
//!
class ISafePluginRegistry
{
public:
    //!
    //! \brief Register a plugin creator.
    //!
    //! \pre INIT API
    //!
    //! \param creator The plugin creator to be registered.
    //!
    //! \param pluginNamespace A NULL-terminated namespace string, which must be 1024 bytes or less including the NULL
    //! terminator. It must be identical with the result of calling
    //! ISafePluginCreatorV3One::getPluginNamespace() on the creator object.
    //!
    //! \param recorder The error recorder to register with this interface
    //!
    //! \return Returns kSUCCESS if the plugin creator is successfully registered.
    //!         Returns kINVALID_ARGUMENT if the pluginNamespace string is a nullptr, exceeds the maximum length,
    //!         or does not match the result of creator.getPluginNamespace().
    //!         Returns kINVALID_CONFIG if there have already been 100 plugin creators registered
    //!         or another plugin creator with the same combination of plugin name, version, and namespace has already
    //!         been registered.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes; calls to this method will be synchronized by a mutex.
    //!
    virtual ErrorCode registerCreator(
        IPluginCreatorInterface& creator, AsciiChar const* const pluginNamespace, ISafeRecorder& recorder) noexcept = 0;

    //!
    //! \brief Return all the registered plugin creators and the number of
    //! registered plugin creators.
    //!
    //! \pre INIT API
    //!
    //! \param[out] creators The start address of an IPluginCreatorInterface* array of length numCreators if at least
    //! one plugin creator
    //!                      has been registered, or nullptr if there are no registered plugin creators.
    //!
    //! \param[out] numCreators If the call completes successfully, the number of registered plugin creators (which
    //!                         will be an integer between 0 and 100 inclusive).
    //!
    //! \return Returns kSUCCESS if the call completes successfully and at least one plugin creator is registered.
    //!         Returns kINVALID_ARGUMENT if numCreators or creators is a nullptr.
    //!         Returns kUNSPECIFIED_ERROR if an unexpected error occurs.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual ErrorCode getAllCreators(
        IPluginCreatorInterface* const*& creators, int32_t& numCreators) const noexcept = 0;

    //!
    //! \brief Return plugin creator based on plugin name, version, and
    //! namespace associated with plugin during network creation.
    //!
    //! \pre INIT API
    //!
    //! \warning The strings pluginName, pluginVersion, and pluginNamespace must be NULL terminated and have a length
    //! of 1024 bytes or less including the NULL terminator.
    //!
    //! \param[out] creator Pointer to the plugin creator pointer to be returned.
    //! \param pluginName The plugin name string.
    //! \param pluginVersion The plugin version string.
    //! \param pluginNamespace The plugin namespace (by default empty string).
    //!
    //! \return Returns kSUCCESS if a plugin creator corresponding to the passed name, version, and namespace can be
    //! found in the
    //!         registry. Returns kINVALID_ARGUMENT if any of the input arguments is nullptr or exceeds the string
    //!         length limit. Returns kINVALID_CONFIG if no plugin creator corresponding to the input arguments can be
    //!         found in the registry or if a plugin creator can be found but its stored namespace attribute does not
    //!         match the pluginNamespace.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual ErrorCode getCreator(IPluginCreatorInterface*& creator, AsciiChar const* const pluginName,
        AsciiChar const* const pluginVersion, AsciiChar const* const pluginNamespace = "") noexcept = 0;

    //!
    //! \brief Set the message recorder for this interface.
    //!
    //! \pre INIT API
    //!
    //! Assigns the message recorder to this interface. The message recorder will track all errors during execution.
    //! This function will call incRefCount of the registered message recorder at least once.
    //!
    //! \note Since the pluginRegistry is a global static object, the recorder used for the registry must have an equal
    //! or longer life time
    //!       than the registry. I.e., when the process terminates the registry needs to hold a global static error
    //!       recorder.
    //!
    //! \param recorder The error recorder to register with this interface, or nullptr to deregister the current
    //!                 recorder.
    //!
    //! \return Returns kSUCCESS if the message recorder is successfully registered or deregistered.
    //!         Returns kINVALID_ARGUMENT if the recorder is set to nullptr and the recorder has been registered.
    //!
    //! \see getSafeRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual ErrorCode setSafeRecorder(ISafeRecorder& recorder) noexcept = 0;

    //!
    //! \brief Get the message recorder assigned to this interface.
    //!
    //! \pre INIT API
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so recorder parameter will be set to nullptr (pointer to nullptr) if setSafeRecorder has not been called,
    //! or an message recorder has not been inherited.
    //!
    //! \param[out] recorder Pointer to the ISafeRecorder object to be returned.
    //!
    //! \return Returns kSUCCESS if the message recorder is successfully retrieved.
    //!         Returns kINVALID_ARGUMENT if the recorder parameter is nullptr.
    //!
    //! \see setSafeRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual ErrorCode getSafeRecorder(ISafeRecorder*& recorder) const noexcept = 0;

    //!
    //! \brief Deregister a previously registered plugin creator.
    //!
    //! \pre INIT API
    //!
    //! Since there may be a desire to limit the number of plugins,
    //! this function provides a mechanism for removing plugin creators registered in TensorRT.
    //! The plugin creator that is specified by \p creator is removed from TensorRT and no longer tracked.
    //!
    //! \param creator The plugin creator to deregister.
    //!
    //! \return Returns kSUCCESS if the plugin creator was successfully deregistered.
    //!         Returns kINVALID_CONFIG if the plugin creator was not found in the registry or otherwise could not be
    //!         deregistered.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual ErrorCode deregisterCreator(IPluginCreatorInterface const& creator) noexcept = 0;

    // @cond SuppressDoxyWarnings
    ISafePluginRegistry() = default;
    ISafePluginRegistry(ISafePluginRegistry const&) = delete;
    ISafePluginRegistry(ISafePluginRegistry&&) = delete;
    ISafePluginRegistry& operator=(ISafePluginRegistry const&) & = delete;
    ISafePluginRegistry& operator=(ISafePluginRegistry&&) & = delete;
    // @endcond

protected:
    virtual ~ISafePluginRegistry() noexcept = default;
};

//!
//! \class IPluginV3OneSafeCore
//!
//! \brief Interface specifying the base plugin capability.
//!
class IPluginV3OneSafeCore : public IPluginCapability
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    //! \pre INIT API
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3ONE_SAFE_CORE", 1, 0};
    }

    //! \brief Copy constructor
    IPluginV3OneSafeCore(IPluginV3OneSafeCore const&) = default;

    //! \brief Move constructor
    IPluginV3OneSafeCore(IPluginV3OneSafeCore&&) = default;

    //! \brief Copy assignment operator
    IPluginV3OneSafeCore& operator=(IPluginV3OneSafeCore const&) & = default;

    //! \brief Move assignment operator
    IPluginV3OneSafeCore& operator=(IPluginV3OneSafeCore&&) & = default;

    //! \brief Destructor for IPluginV3OneSafeCore.
    ~IPluginV3OneSafeCore() noexcept override = default;

    //!
    //! \brief Return the plugin name. Should match the plugin name returned by the corresponding plugin creator.
    //!
    //! \pre INIT API
    //!
    //! \see ISafePluginCreatorV3One::getPluginName()
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    virtual AsciiChar const* getPluginName() const noexcept = 0;

    //!
    //! \brief Return the plugin version. Should match the plugin version returned by the corresponding plugin creator.
    //!
    //! \pre INIT API
    //!
    //! \see ISafePluginCreatorV3One::getPluginVersion()
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    //!
    //! \brief Return the namespace of the plugin object. Should match the plugin namespace returned by the
    //! corresponding plugin creator.
    //!
    //! \pre INIT API
    //!
    //! \see ISafePluginCreatorV3One::getPluginNamespace()
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;

protected:
    //! \brief Default constructor.
    IPluginV3OneSafeCore() = default;
};

class IPluginV3OneSafeBuild : public IPluginCapability
{
public:
    //!
    //! \brief The default maximum number of format combinations that will be timed by TensorRT during the build phase
    //!
    //! \see getFormatCombinationLimit
    //!
    static constexpr int32_t kDEFAULT_FORMAT_COMBINATION_LIMIT = 100;

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3ONE_SAFE_BUILD", 1, 0};
    }

    IPluginV3OneSafeBuild() = default;
    IPluginV3OneSafeBuild(IPluginV3OneSafeBuild&&) = delete;
    IPluginV3OneSafeBuild& operator=(IPluginV3OneSafeBuild&&) & = delete;
    //! \brief Destructor for IPluginV3OneSafeBuild.
    ~IPluginV3OneSafeBuild() noexcept override = default;

    //!
    //! \brief Configure the plugin.
    //!
    //! configurePlugin() can be called multiple times in the build phase during creation of an engine by IBuilder.
    //!
    //! configurePlugin() is called when a plugin is being prepared for profiling but not for any
    //! specific input size. This provides an opportunity for the plugin to make algorithmic choices on the basis of
    //! input and output formats, along with the bound of possible dimensions. The min, opt and max value of the
    //! TensorDescriptor correspond to the kMIN, kOPT and kMAX value of the current profile that the plugin is
    //! being profiled for, with the desc.dims field corresponding to the dimensions of plugin specified at network
    //! creation. Wildcard dimensions may exist during this phase in the desc.dims field.
    //!
    //! \param in The input tensors attributes that are used for configuration.
    //! \param nbInputs Number of input tensors.
    //! \param out The output tensors attributes that are used for configuration.
    //! \param nbOutputs Number of output tensors.
    //!
    virtual int32_t configurePlugin(
        TensorDescriptor const* in, int32_t nbInputs, TensorDescriptor const* out, int32_t nbOutputs) noexcept = 0;

    //!
    //! \brief Provide the data types of the plugin outputs if the input tensors have the data types provided.
    //!
    //! \param outputTypes Pre-allocated array to which the output data types should be written.
    //! \param nbOutputs The number of output tensors. This matches the value returned from getNbOutputs().
    //! \param inputTypes The input data types.
    //! \param nbInputs The number of input tensors.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). The returned code will be reported
    //! through the error recorder.
    //!
    //! \note Provide `DataType::kFLOAT`s if the layer has no inputs. The data type for any size tensor outputs must be
    //! `DataType::kINT32`. The returned data types must each have a format that is supported by the plugin.
    //!
    //! \warning DataType:kBOOL and DataType::kUINT8 are not supported.
    //!
    virtual int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
        = 0;

    //!
    //! \brief Provide expressions for computing dimensions of the output tensors from dimensions of the input tensors.
    //!
    //! \param inputs Expressions for dimensions of the input tensors
    //! \param nbInputs The number of input tensors
    //! \param shapeInputs Expressions for values of the shape tensor inputs
    //! \param nbShapeInputs The number of shape tensor inputs
    //! \param outputs Pre-allocated array to which the output dimensions must be written
    //! \param exprBuilder Object for generating new dimension expressions
    //!
    //! \note Any size tensor outputs must be declared to be 0-D.
    //!
    //! \note The declaration of shapeInputs as DimsExprs is slightly abusive, because the "dimensions"
    //!       are actually the values of the shape tensor. For example, if the input shape tensor
    //!       is a 2x3 matrix, the DimsExprs will have six "dimensions": the three values from the first
    //!       row of the matrix followed by the three values from the second row of the matrix.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). Returned code will be reported
    //! through the error recorder.
    //!
    virtual int32_t getOutputShapes(
        Dims const* inputs, int32_t nbInputs, Dims* outputs, int32_t nbOutputs) const noexcept = 0;

    //!
    //! \brief Return true if plugin supports the format and datatype for the input/output indexed by pos.
    //!
    //! For this method inputs are numbered 0.. (nbInputs - 1) and outputs are numbered nbInputs.. (nbInputs + nbOutputs
    //! - 1). Using this numbering, pos is an index into InOut, where 0 <= pos < nbInputs + nbOutputs - 1.
    //!
    //! TensorRT invokes this method to ask if the input/output indexed by pos supports the format/datatype specified
    //! by inOut[pos].format and inOut[pos].type.  The override should return true if that format/datatype at inOut[pos]
    //! are supported by the plugin.  If support is conditional on other input/output formats/datatypes, the plugin can
    //! make its result conditional on the formats/datatypes in inOut[0.. pos - 1], which will be set to values
    //! that the plugin supports.  The override should not inspect inOut[pos1.. nbInputs + nbOutputs - 1],
    //! which will have invalid values.  In other words, the decision for pos must be based on inOut[0..pos] only.
    //!
    //! Some examples:
    //!
    //! * A definition for a plugin that supports only FP16 NCHW:
    //!
    //!         return inOut.format[pos] == TensorFormat::kLINEAR && inOut.type[pos] == DataType::kHALF;
    //!
    //! * A definition for a plugin that supports only FP16 NCHW for its two inputs,
    //!   and FP32 NCHW for its single output:
    //!
    //!         return inOut.format[pos] == TensorFormat::kLINEAR && (inOut.type[pos] == pos < 2 ?  DataType::kHALF :
    //!         DataType::kFLOAT);
    //!
    //! * A definition for a "polymorphic" plugin with two inputs and one output that supports
    //!   any format or type, but the inputs and output must have the same format and type:
    //!
    //!         return pos == 0 || (inOut.format[pos] == inOut.format[0] && inOut.type[pos] == inOut.type[0]);
    //!
    //! \warning TensorRT will stop querying once it finds getFormatCombinationLimit() of combinations.
    //!
    //! \see getFormatCombinationLimit
    //!
    virtual bool supportsFormatCombination(
        int32_t pos, TensorDescriptor const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept = 0;

    //!
    //! \brief Get the number of outputs from the plugin.
    //!
    //! \return The number of outputs, which must be a positive integer.
    //!
    virtual int32_t getNbOutputs() const noexcept = 0;

    //!
    //! \brief Find the workspace size required by the layer.
    //!
    //! This function is called after the plugin is configured, and possibly during execution.
    //! The result should be a sufficient workspace size to deal with inputs and outputs of the given size
    //! or any smaller problem.
    //!
    //! \return The workspace size.
    //!
    virtual size_t getWorkspaceSize(TensorDescriptor const* /*inputs*/, int32_t /*nbInputs*/,
        TensorDescriptor const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept
    {
        return 0U;
    }

    //!
    //! \brief Query for any custom tactics that the plugin intends to use
    //!
    //! This method queries for the set of tactics T(f) supported by the plugin for the format combination f indicated
    //! by the immediately preceding call to configurePlugin(). It is guaranteed to be called after configurePlugin().
    //!
    //! For each format combination provided through configurePlugin(), up to a maximum of getFormatCombinationLimit(),
    //! the plugin will be timed for each tactic advertised through this method for that format combination. i.e. The
    //! plugin will be timed \f$N = sum_{i=0}^{i<getFormatCombinationLimit()} (T(f[i]))\f$ times. If \f$N = 1\f$, the
    //! plugin may not be timed. In peudocode, the timing protocol appears as the following:
    //!
    //! counter = 0
    //! for each supported format combination
    //!     ++counter
    //!     if counter > getFormatCombinationLimit()
    //!         goto done
    //!     configurePlugin(...)
    //!     for each tactic in getValidTactics(...)
    //!         time tactic
    //! done:
    //!
    //!
    //! \param tactics Pre-allocated buffer to which the tactic values should be written
    //! \param nbTactics The number of tactics advertised through getNbTactics()
    //!
    //! \note The provided tactic values must be unique and non-zero. The tactic value 0 is reserved for the default
    //! tactic attached to each format combination.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). The returned code will be reported
    //! through the error recorder.
    //!
    virtual int32_t getValidTactics(int32_t* /*tactics*/, int32_t /*nbTactics*/) noexcept
    {
        return 0;
    }

    //!
    //! \brief Query for the number of custom tactics the plugin intends to use
    //!
    virtual int32_t getNbTactics() noexcept
    {
        return 0;
    }

    //!
    //! \brief Called to query the suffix to use for the timing cache ID. May be called anytime after plugin creation.
    //!
    //! \return Suffix to use for timing cache ID, considering only the creation state of the plugin.
    //!         Returning nullptr will disable timing caching for the plugin altogether.
    //!
    //! \note If timing caching is enabled for the plugin (by returning non-null), the I/O shape and format information
    //! will be automatically considered to form the prefix of the timing cache ID. Therefore, only other factors
    //! determining the creation state of the plugin, such as its attribute values, should be considered to compose the
    //! return value.
    //!
    virtual char const* getTimingCacheID() noexcept
    {
        return nullptr;
    }

    //!
    //! \brief Return the maximum number of format combinations that will be timed by TensorRT during the build phase
    //!
    virtual int32_t getFormatCombinationLimit() noexcept
    {
        return kDEFAULT_FORMAT_COMBINATION_LIMIT;
    }

    //!
    //! \brief Query for a string representing the configuration of the plugin. May be called anytime after
    //! plugin creation.
    //!
    //! \return A string representing the plugin's creation state, especially with regard to its attribute values.
    //!
    virtual char const* getMetadataString() noexcept
    {
        return nullptr;
    }

protected:
    IPluginV3OneSafeBuild(IPluginV3OneSafeBuild const&) = default;
    IPluginV3OneSafeBuild& operator=(IPluginV3OneSafeBuild const&) & = default;
};

//!
//! \class IPluginV3OneSafeBuildMSS
//!
//! \brief This class provides build capability and let output shapes be calculated by symbolic expression
//! to support multiple static shape.
//!
class IPluginV3OneSafeBuildMSS : public IPluginV3OneSafeBuild
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3ONE_SAFE_BUILDMSS", 1, 0};
    }

    int32_t getOutputShapes(
        Dims const* /*inputs*/, int32_t /*nbInputs*/, Dims* /*outputs*/, int32_t /*nbOutputs*/) const noexcept override
    {
        // This function will not be used, instead, getSymbolicOutputShapes should be used.
        return 0;
    }

    //! \brief Copy constructor
    IPluginV3OneSafeBuildMSS(IPluginV3OneSafeBuildMSS const&) = default;

    //! \brief Move constructor
    IPluginV3OneSafeBuildMSS(IPluginV3OneSafeBuildMSS&&) = delete;

    //! \brief Copy assignment operator
    IPluginV3OneSafeBuildMSS& operator=(IPluginV3OneSafeBuildMSS const&) & = default;

    //! \brief Move assignment operator
    IPluginV3OneSafeBuildMSS& operator=(IPluginV3OneSafeBuildMSS&&) & = delete;

    //! \brief Destructor for IPluginV3OneSafeBuildMSS.
    ~IPluginV3OneSafeBuildMSS() noexcept override = default;
    //!
    //! \brief Provide expressions for computing dimensions of the output tensors from dimensions of the input tensors.
    //!
    //! \param inputs Expressions for dimensions of the input tensors
    //! \param nbInputs The number of input tensors
    //! \param outputs Pre-allocated array to which the output dimensions must be written
    //! \param nbOutputs The number of output tensors
    //! \param exprBuilder Object for generating new dimension expressions
    //!
    //! \note Any size tensor outputs must be declared to be 0-D.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). Returned code will be reported
    //! through the error recorder.
    //!
    virtual int32_t getSymbolicOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) const noexcept = 0;

protected:
    //! \brief Default constructor.
    IPluginV3OneSafeBuildMSS() = default;
};

//!
//! \class IPluginV3OneSafeRuntime
//
//! \brief Runtime component of the pluginV3 system, this component needs to be safety certified and is required at
//! runtime.
//!
class IPluginV3OneSafeRuntime : public IPluginCapability
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    //! \pre INIT API
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3ONE_SAFE_RUNTIME", 1, 0};
    }

    //! \brief Copy constructor
    IPluginV3OneSafeRuntime(IPluginV3OneSafeRuntime const&) = default;

    //! \brief Move constructor
    IPluginV3OneSafeRuntime(IPluginV3OneSafeRuntime&&) = default;

    //! \brief Copy assignment operator
    IPluginV3OneSafeRuntime& operator=(IPluginV3OneSafeRuntime const&) & = default;

    //! \brief Move assignment operator
    IPluginV3OneSafeRuntime& operator=(IPluginV3OneSafeRuntime&&) & = default;

    //! \brief Destructor for IPluginV3OneSafeRuntime.
    ~IPluginV3OneSafeRuntime() noexcept override = default;

    //!
    //! \brief Set the tactic to be used in the subsequent call to enqueue(). If no custom tactics were advertised, this
    //! will have a value of 0, which is designated as the default tactic.
    //!
    //! \pre INIT API
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). The returned code will be reported
    //! through the error recorder.
    //!
    virtual int32_t setTactic(int32_t /*tactic*/) noexcept
    {
        return 0;
    }

    //!
    //! \brief Execute the layer.
    //!
    //! \pre RUNTIME API
    //!
    //! \param inputDesc how to interpret the memory for the input tensors.
    //! \param outputDesc how to interpret the memory for the output tensors.
    //! \param inputs The memory for the input tensors.
    //! \param outputs The memory for the output tensors.
    //! \param workspace Workspace for execution.
    //! \param stream The stream in which to execute the kernels.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). The returned code will be reported
    //! through the error recorder.
    //!
    virtual int32_t enqueue(TensorDescriptor const* inputDesc, TensorDescriptor const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept = 0;

    //!
    //! \brief This function will be called to initialize a plugin object after it is created (before execution phase),
    //! either during graph deserialization or when the graph is cloned. The overriding implementations of this
    //! function can use the provided context object to access the message recorder, the memory allocator and the
    //! runtime error buffer of the graph context to which this plugin belongs.
    //!
    //! \pre INIT API
    //!
    //! \param ISafePluginResourceContext A pointer to the ISafePluginResourceContext which contains the message
    //! recorder, the memory allocator, and the runtime error information buffer.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). The returned code will be reported
    //! through the error recorder.
    //!
    virtual int32_t initResource(ISafePluginResourceContext const* context) noexcept = 0;

    //!
    //! \brief Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with
    //! these parameters. This call will always be followed by a call to initResource() to complete any initializations
    //! that depend on the graph context.
    //!
    //! \pre INIT API
    //!
    virtual IPluginV3* clone() noexcept = 0;

    //!
    //! \brief Get the plugin fields which should be serialized.
    //!
    //! \pre INIT API
    //!
    //! \note The set of plugin fields returned does not necessarily need to match that advertised through
    //! getFieldNames() of the corresponding plugin creator.

    //! \note To serialize arbitrary plugin data, use a PluginField of
    //! PluginFieldType::kUNKNOWN, with the length of the PluginField set to the correct number of bytes.
    //!
    virtual PluginFieldCollection const* getFieldsToSerialize() noexcept = 0;

protected:
    //! \brief Default constructor.
    IPluginV3OneSafeRuntime() = default;
};

//!
//! \class ISafePluginCreatorV3One
//
//! \brief The main interface to be implemented by all plugin creator classes.
//!
class ISafePluginCreatorV3One : public IPluginCreatorInterface
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    //! \pre INIT API
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN SAFE CREATOR_V3ONE", 1, 0};
    }

    //!
    //! \brief Return a plugin object. Return nullptr in case of error.
    //!
    //! \pre INIT API
    //!
    //! \param name A NULL-terminated name string of length 1024 or less, including the NULL terminator.
    //! \param fc A pointer to a collection of fields needed for constructing the plugin.
    //! \param phase The TensorRT phase in which the plugin is being created
    //!
    //! When the phase is TensorRTPhase::kRUNTIME, the PluginFieldCollection provided for serialization by the plugin's
    //! runtime interface will be passed as fc.
    //!
    //! \note The returned plugin object must be in an initialized state
    //!
    virtual IPluginV3* createPlugin(
        AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept = 0;

    //!
    //! \brief Return a list of fields that need to be passed to createPlugin() when creating a plugin for use in the
    //! TensorRT build phase.
    //!
    //! \pre INIT API
    //!
    //! \see PluginFieldCollection
    //!
    virtual PluginFieldCollection const* getFieldNames() noexcept = 0;

    //!
    //! \brief Return the plugin name.
    //!
    //! \pre INIT API
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including
    //! the NULL terminator.
    //!
    virtual AsciiChar const* getPluginName() const noexcept = 0;

    //!
    //! \brief Return the plugin version.
    //!
    //! \pre INIT API
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including
    //! the NULL terminator.
    //!
    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    //!
    //! \brief Return the plugin namespace.
    //!
    //! \pre INIT API
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including
    //! the NULL terminator.
    //!
    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;

    //!
    //! \brief Return the safe recorder
    //!
    //! \pre INIT API
    //!
    virtual ISafeRecorder* getSafeRecorder() const noexcept = 0;

    //!
    //! \brief Set the safe recorder for this interface.
    //!
    //! \pre INIT API
    //!
    virtual void setSafeRecorder(ISafeRecorder&) noexcept = 0;

    ISafePluginCreatorV3One() = default;
    ~ISafePluginCreatorV3One() noexcept override = default;

protected:
    ISafePluginCreatorV3One(ISafePluginCreatorV3One const&) = default;
    ISafePluginCreatorV3One(ISafePluginCreatorV3One&&) = default;
    ISafePluginCreatorV3One& operator=(ISafePluginCreatorV3One const&) & = default;
    ISafePluginCreatorV3One& operator=(ISafePluginCreatorV3One&&) & = default;
};

//!
//! \brief Get the SafePluginRegistry singleton instance and set the SafeRecorder of the safePluginRegistry.
//!
//! \pre INIT API
//!
//! This function retrieves the singleton instance of the SafePluginRegistry and sets the provided SafeRecorder
//! for error tracking. The SafePluginRegistry is used to manage plugin creators in a safe and thread-safe manner.
//!
//! \param recorder The SafeRecorder to be set for the SafePluginRegistry instance.
//!
//! \return A pointer to the ISafePluginRegistry interface of the SafePluginRegistry instance.
//!
//! \usage
//! - Allowed context for the API call
//!   - Thread-safe: Yes
//!
extern "C" nvinfer2::safe::ISafePluginRegistry* getSafePluginRegistry(nvinfer2::safe::ISafeRecorder& recorder) noexcept;
} // namespace safe
} // namespace nvinfer2

#endif /* NV_INFER_SAFE_PLUGIN_H */
