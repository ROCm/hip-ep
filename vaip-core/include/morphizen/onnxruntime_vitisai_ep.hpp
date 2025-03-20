/*
 *  Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 **/
#pragma once
#include <filesystem>
#include <vector>
#include <vaip/export.h>
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
