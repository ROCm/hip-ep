/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <filesystem>
#include <vaip/export.h>
#include <vector>
namespace vaip_core {
struct OrtApiForVaip;
} // namespace vaip_core

namespace onnxruntime_vitisai_ep {
VAIP_DLL_SPEC
int optimize_onnx_model(const std::filesystem::path& model_path_in,
                        const std::filesystem::path& model_path_out,
                        const char* json_config);
VAIP_DLL_SPEC
void initialize_graph_optimizer(const std::string& json_path);
} // namespace onnxruntime_vitisai_ep

extern "C" {
VAIP_DLL_SPEC const vaip_core::OrtApiForVaip* get_the_global_api();
}
