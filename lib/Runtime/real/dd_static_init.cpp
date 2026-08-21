/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Define missing static members from DynamicDispatch xrt_context class
#include <xrt_context/xrt_context.hpp>

namespace ryzenai {
namespace dynamic_dispatch {

// Define the missing static member
// This matches the declaration in xrt_context.hpp
std::unordered_map<std::string, std::shared_ptr<xrt_context>> *xrt_context::ctx_map_p_ = nullptr;

} // namespace dynamic_dispatch
} // namespace ryzenai
