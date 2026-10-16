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

#ifndef NV_INFER_REFERENCE_CHECKER_H
#define NV_INFER_REFERENCE_CHECKER_H
#include "NvInferForwardDecl.h"
#include "NvInferSafeRuntime.h"
#include <cstdint>

//!
//! \file NvInferReference.h
//!
//! Per-kernel numerical reference check for a safety-certified TRT engine.
//!
//! The check re-derives each generated kernel's boundary outputs on the CPU and compares them against
//! the kernel run on a remote target.
//!
//! Diagnostics are surfaced through the ISafeRecorder, and validate() returns a single pass/fail.
//!

namespace nvinfer2
{
namespace safe
{
// Forward declaration
class ISafeRecorder;

namespace reference
{

//!
//! \class IReferenceChecker
//!
//! \brief Validates a safety-certified TRT engine (serialized binary blob) by re-deriving each
//!        generated kernel's boundary outputs on CPU and comparing them against the kernel run on
//!        the remote target.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//!          and ABI.
//!
class IReferenceChecker
{
public:
    //!
    //! \brief Run the reference check over the engine blob that was input to createReferenceChecker.
    //!
    //! For each shipping generated kernel described by the engine, the checker builds synthetic
    //! boundary inputs, computes a golden boundary output locally on CPU, runs the kernel on the
    //! remote target, and compares the two. Per-kernel/per-pattern diagnostics are reported through
    //! the ISafeRecorder supplied to the factory.
    //!
    //! \return true if every checked kernel matched its golden reference within tolerance, false
    //!         otherwise (including transport/setup failures, which are reported via the recorder).
    //!
    virtual bool validate() noexcept = 0;
    //!
    //! \brief De-allocates any internally allocated memory.
    //!
    virtual ~IReferenceChecker() = default;

protected:
    //!
    //! \brief Default constructor for IReferenceChecker.
    //! \details Constructs an object in a default state. Access is protected; only derived types or
    //!          the factory createReferenceChecker() may create instances. The object does not yet
    //!          hold or validate any blob until initialized by the factory.
    //! \pre none
    //! \post Object is in default-constructed state and is not yet bound to any engine blob.
    //!
    IReferenceChecker() = default;
    //!
    //! \brief Deleted copy constructor to maintain non-copyability.
    //! \pre none
    //! \post none
    //!
    IReferenceChecker(IReferenceChecker const& other) = delete;
    //!
    //! \brief Deleted copy assignment operator (lvalue) to maintain non-copyability.
    //! \pre none
    //! \post none
    //!
    IReferenceChecker& operator=(IReferenceChecker const& other) & = delete;
    //!
    //! \brief Deleted move constructor to maintain non-movability.
    //! \pre none
    //! \post none
    //!
    IReferenceChecker(IReferenceChecker&& other) = delete;
    //!
    //! \brief Deleted move assignment operator (lvalue) to maintain non-movability.
    //! \pre none
    //! \post none
    //!
    IReferenceChecker& operator=(IReferenceChecker&& other) & = delete;
};

//!
//! \brief Create an instance of an IReferenceChecker class.
//!
//! ISafeRecorder is the error recorder class for the reference checker.
//!
//! \param irc          On success, set to a newly-allocated reference checker owned by the caller.
//! \param recorder     Recorder for diagnostics.
//! \param engineBlob   The serialized safe engine to validate.
//! \param engineBlobSize Size of \p engineBlob in bytes. Must be positive.
//! \param checkerBlob   The checker blob the compiler wrote beside the engine (trtexec
//!                     --dumpCheckerBlob, `<engine>.txt`), which carries the per-kernel metadata a
//!                     golden is derived from. May be nullptr: an engine whose instructions are all
//!                     static library kernels is out of scope for a reference model and needs none,
//!                     and validate() reports the omission for any engine that does need it.
//! \param checkerBlobSize Size of \p checkerBlob in bytes. Must be 0 exactly when \p checkerBlob is
//!                     nullptr.
//! \param remoteConfig Opaque connection token for the remote target that runs the engine's kernels,
//!                     in ssh://user[:pass]@host[:port]?remote_exec_path=<path>[&remote_lib_path=<path>]
//!                     form. Must not be empty: remote is the only supported target location.
//!
extern "C" nvinfer2::safe::ErrorCode createReferenceChecker(IReferenceChecker*& irc,
    nvinfer2::safe::ISafeRecorder& recorder, void const* engineBlob, int64_t engineBlobSize, void const* checkerBlob,
    int64_t checkerBlobSize, char const* remoteConfig) noexcept;

} // namespace reference

} // namespace safe
} // namespace nvinfer2

#endif // NV_INFER_REFERENCE_CHECKER_H
