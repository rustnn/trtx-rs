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

// This file provides the user-implementable interface for the Safe Recorder feature in the NVIDIA Safe Runtime API.
// The Safe Recorder is a user-defined object that interacts with the internal state of the object it is assigned to,
// in order to determine information about abnormalities in execution.
// The Safe Recorder interface, ISafeRecorder, allows users to define a custom error reporting mechanism that receives
// both an error enum and a string description of the error. The error strings are limited to 255 bytes or less in
// length, including the NULL terminator. The Safe Recorder is passed along to any class that is created from another
// class that has an Safe Recorder assigned to it, allowing for hierarchical error reporting. Users can register a
// different error recorder or de-register the error recorder for specific objects using accessor functions. The Safe
// Recorder object implementation must be thread-safe, with all locking and synchronization handled by the interface
// implementation. The lifetime of the Safe Recorder object must exceed the lifetime of all TensorRT objects that use
// it.

#ifndef NV_INFER_SAFE_RECORDER_H
#define NV_INFER_SAFE_RECORDER_H

#include "NvInferForwardDecl.h"
#include <cstdint>

namespace nvinfer2
{
namespace safe
{

using ErrorDesc = char const*; //!< Type alias for error description strings.
using RefCount = int32_t;      //!< Type alias for reference count.

//!
//! \enum Severity
//! \brief Enumerates severity levels for messages issued by the message recorder.
//!
enum class Severity : uint32_t
{
    kINTERNAL_ERROR = 0, //!< An internal error has occurred. Execution is unrecoverable.
    kERROR = 1,          //!< An application error has occurred.
    kWARNING = 2, //!< An application error has been discovered, but TensorRT has recovered or fallen back to a default.
    kINFO = 3,    //!< Informational messages with instructional information.
    kVERBOSE = 4, //!< Verbose messages with additional information.
    kDEBUG = 5,   //!< Trace messages with debugging information.
};

namespace impl
{
//! \brief Provides a static constant for the maximum value of the Severity enum.
template <>
struct EnumMaxImpl<Severity>
{
    static constexpr int32_t kVALUE = 6; //!< Maximum number of elements in the Severity enum.
};
} // namespace impl

//!
//! \class ISafeRecorder
//! \brief Interface for extended recorder which allows error, warn, debug, or info messages to be recorded.
//! \details Inherits from IErrorRecorder and extends functionality to include various levels of message severity.
//! \note Any user implementation of this interface must ensure that error messages are always reported and not
//! suppressed due to lower reporting severity.
//!
class ISafeRecorder : public IErrorRecorder
{
public:
    //!
    //! \brief The length limit for an error description in bytes, excluding the '\0' string terminator.
    //! \note  Only applicable to messages issued by the TensorRT safe runtime.
    //!        User recorder implementation can use any size appropriate for the use case.
    //!
    static constexpr size_t kMAX_SAFE_DESC_LENGTH{4095U};

    //! \brief Constructor for ISafeRecorder.
    //!
    //! \pre INIT API
    //!
    //! \param[in] severity_ Initial severity level for recording messages.
    //! Only messages with severity less than or equal to this severity level will be reported.
    //! \param[in] id_ Optional identifier for the recorder instance. This can be used to
    //! identify messages issued by different recorders in multiple threads.
    ISafeRecorder(Severity severity_ = Severity::kERROR, int32_t id_ = -1)
        : IErrorRecorder()
        , mSeverity(severity_)
        , mId(id_)
    {
    }

    ISafeRecorder(ISafeRecorder const&) = delete;
    ISafeRecorder(ISafeRecorder&&) = delete;
    ISafeRecorder& operator=(ISafeRecorder const&) & = delete;
    ISafeRecorder& operator=(ISafeRecorder&&) & = delete;

    //! \brief Destructor for ISafeRecorder.
    ~ISafeRecorder() noexcept override = default;

    //! \brief Sets the severity level for the recorder.
    //!
    //! \pre RUNTIME API
    //!
    //! \param[in] severity_ New severity level for recording messages.
    //!
    //! \remark Not thread-safe
    virtual void setSeverity(Severity severity_) noexcept
    {
        mSeverity = severity_;
    }
    // NOLINTBEGIN to avoid clang-tidy requiring [[nodiscard]]
    //! \brief Get the current severity level of the recorder.
    //!
    //! \pre RUNTIME API
    //!
    //! \return The severity level of the recorder.
    //!
    //! \remark This method is required to be thread-safe and may be called from multiple threads
    //!         when multiple execution contexts are used during runtime.
    virtual Severity getSeverity() const noexcept
    {
        return mSeverity;
    }

    //! \brief Gets the identifier of the recorder.
    //!
    //! \pre RUNTIME API
    //!
    //! \return The identifier of the recorder.
    //!
    //! \remark This method is required to be thread-safe and may be called from multiple threads
    //!         when multiple execution contexts are used during runtime.
    virtual int32_t getId() const noexcept
    {
        return mId;
    }
    // NOLINTEND

    //! \brief Reports a warning message.
    //!
    //! \pre RUNTIME API
    //!
    //! \param[in] desc Description of the warning.
    //! \return True if the message was reported successfully, false otherwise.
    //!
    //! \remark This method is required to be thread-safe and may be called from multiple threads
    //!         when multiple execution contexts are used during runtime.
    virtual bool reportWarn(ErrorDesc /*desc*/) noexcept
    {
        return false;
    }

    //! \brief Reports a debug message.
    //!
    //! \pre RUNTIME API
    //!
    //! \param[in] desc Description of the debug information.
    //! \return True if the message was reported successfully, false otherwise.
    //!
    //! \remark This method is required to be thread-safe and may be called from multiple threads
    //!         when multiple execution contexts are used during runtime.
    virtual bool reportDebug(ErrorDesc /*desc*/) noexcept
    {
        return false;
    }

    //! \brief Reports an informational message.
    //!
    //! \pre RUNTIME API
    //!
    //! \param[in] desc Description of the information.
    //! \return True if the message was reported successfully, false otherwise.
    //!
    //! \remark This method is required to be thread-safe and may be called from multiple threads
    //!         when multiple execution contexts are used during runtime.
    virtual bool reportInfo(ErrorDesc /*desc*/) noexcept
    {
        return false;
    }

    //! \brief Reports a verbose informational  message.
    //!
    //! \pre RUNTIME API
    //!
    //! \param[in] desc Description of the information.
    //! \return True if the message was reported successfully, false otherwise.
    //!
    //! \remark This method is required to be thread-safe and may be called from multiple threads
    //!         when multiple execution contexts are used during runtime.
    virtual bool reportVerbose(ErrorDesc /*desc*/) noexcept
    {
        return false;
    }

protected:
    //! Severity level of the recorder.
    Severity mSeverity;

    //! ID of the recorder.
    int32_t const mId;
};

//! \struct RuntimeErrorInformation
//! \brief Holds information about runtime errors that occur during asynchronous kernel execution.
//! \details Stores a bitmask where each bit represents a different type of runtime error.
//! \usage
//! - Each bit corresponds to a value in \ref RuntimeErrorType.
//! - Set a bit: bitMask |= static_cast<uint64_t>(RuntimeErrorType::<ERROR_NAME>).
//! - Check a bit: (bitMask & static_cast<uint64_t>(RuntimeErrorType::<ERROR_NAME>)) != 0ULL.
//! - Clear a bit: bitMask &= ~static_cast<uint64_t>(RuntimeErrorType::<ERROR_NAME>).
//! - Combine bits: use bitwise OR of multiple RuntimeErrorType values.
struct RuntimeErrorInformation
{
    uint64_t bitMask{0ULL}; //!< Bitmask of errors; see \ref RuntimeErrorType for bit positions.
};

//! \enum RuntimeErrorType
//! \brief Enumerates types of runtime errors that can occur during kernel execution.
//! \details
//! - kNAN_CONSUMED error occurs when a NAN value is stored in an INT8 quantized datatype.
//! - kINF_CONSUMED error occurs when a +/-INF value is stored in an INT8 quantized datatype.
//! - kGATHER_OOB error occurs when a gather index tensor contains a value that is outside of the data tensor.
//! - kSCATTER_OOB error occurs when a scatter index tensor contains a value that is outside of the data tensor.
//! - kSCATTER_RACE error occurs when a scatter index tensor contains duplicate indices with reduction mode kNONE.
//! - kDIV_ZERO error occurs when a division-by-zero happens and its output is of an integer type.
enum class RuntimeErrorType : uint64_t
{
    //! NaN floating-point value was silently consumed
    kNAN_CONSUMED = 1ULL << 0,
    //! Inf floating-point value was silently consumed
    kINF_CONSUMED = 1ULL << 1,
    //! Out-of-bounds access in gather operation
    kGATHER_OOB = 1ULL << 2,
    //! Out-of-bounds access in scatter operation
    kSCATTER_OOB = 1ULL << 3,
    //! Race condition in scatter operation
    kSCATTER_RACE = 1ULL << 4,
    //! Division-by-zero in int division
    kDIV_ZERO = 1ULL << 5,
};

} // namespace safe
} // namespace nvinfer2

#endif /* NV_INFER_SAFE_RECORDER_H */
