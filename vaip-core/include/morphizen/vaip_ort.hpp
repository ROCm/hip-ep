/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "./_sanity_check.hpp"
#if __has_include(<filesystem>)  && __cplusplus > 201700
#  include <filesystem>
#else
#  error "must enable c++17"
#endif

#include <morphizen/vaip-ort-api-ext.hpp>
#include <vaip/custom_op.h>
#include <vaip/export.h>
/// header file used by ort VITISAI execution providers.

using EventInfo = std::tuple<std::string, // name
                             int,         // pid
                             int,         // tid
                             long long,   // timestamp
                             long long    // duration
                             >;

namespace onnxruntime {
using ProviderOptions = std::unordered_map<std::string, std::string>;
}

// Forward declarations for logger types (outside vaip_core namespace)
namespace Ort {
struct Logger;
}

namespace vaip_core {
class PassContextImp;
class LoggerAdapter;

using vaip_error_report_func = void (*)(
    void*, int, const char*); // should be same as vaip_core::error_report_func,
                              // for compile issue we  defined it

VAIP_DLL_SPEC void set_the_global_api(OrtApiForVaip* api);

VAIP_DLL_SPEC const OrtApiForVaip* api();

VAIP_DLL_SPEC std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_5(const std::filesystem::path& model_path,
                     const Graph& graph,
                     const onnxruntime::ProviderOptions& options, void* status,
                     vaip_error_report_func func);

VAIP_DLL_SPEC std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_3(const std::string& model_path, const Graph& graph,
                     const onnxruntime::ProviderOptions& options);

// Internal version that accepts logger_adapter for lifetime management
VAIP_DLL_SPEC std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_3_internal(
    const std::string& model_path, const Graph& onnx_graph,
    const onnxruntime::ProviderOptions& options,
    std::shared_ptr<::Ort::Logger> logger = nullptr,
    std::shared_ptr<LoggerAdapter> logger_adapter = nullptr);

VAIP_DLL_SPEC void profiler_collect(std::vector<EventInfo>& api_events,
                                    std::vector<EventInfo>& kernel_events);

int optimize_onnx_model(const std::filesystem::path& model_path_in,
                        const std::filesystem::path& model_path_out,
                        const char* json_config);

void initialize_graph_optimizer(const std::string& json_path);

} // namespace vaip_core
