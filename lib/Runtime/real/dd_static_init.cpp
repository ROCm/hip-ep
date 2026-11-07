/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// WORKAROUND: Define missing static members from DynamicDispatch xrt_context class
//
// PROBLEM: The DynamicDispatch library declares static members in its headers but
// may not provide their definitions, causing unresolved external linker errors.
// hip-ep cannot reliably know which statics need instantiation without:
//   1. A version number from DD to select the correct set of statics, OR
//   2. DD library itself providing these definitions
//
// CURRENT STATUS: The static member name in xrt_context.hpp is "ctx_map_" (not "ctx_map_p_").
// This definition is disabled (#if 0) until:
//   - DD library provides version info for conditional compilation, OR
//   - DD library fixes their static member definitions, OR
//   - We determine the correct set of statics for the current DD version
//
// If you encounter "unresolved external" errors for xrt_context static members,
// uncomment and fix the definition below to match the actual header declaration.

#include <xrt_context/xrt_context.hpp>

namespace ryzenai {
namespace dynamic_dispatch {

#if 0
// FIXME: Update this definition to match the actual static member name in xrt_context.hpp
// Based on inspection of xrt_context.hpp, the member appears to be named "ctx_map_"
// but this needs verification against the specific DD version being used.
//
// Candidate definition (UNVERIFIED):
// std::unordered_map<std::string, std::shared_ptr<xrt_context>> *xrt_context::ctx_map_ = nullptr;
std::unordered_map<std::string, std::shared_ptr<xrt_context>> *xrt_context::ctx_map_p_ = nullptr;
#endif

} // namespace dynamic_dispatch
} // namespace ryzenai
