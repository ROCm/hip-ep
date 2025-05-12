/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <functional>
#include <string>
namespace vaip_core {
// NOTE: must not add VAIP_DLL_SPEC
// this function cannot be shared between DLLs
void add_cleanup_function(const std::string& name,
                          std::function<void()> cleanup_function);
void deinitialize_onnxruntime_vitisai_ep();
} // namespace vaip_core
