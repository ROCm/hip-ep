/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "morphizen/_sanity_check.hpp"
#if __has_include(<filesystem>)  && __cplusplus > 201700
#  include <filesystem>
#else
#  error "must enable c++17"
#endif

#include <functional>
#include <map>
#include <morphizen/custom_op.h>
#include <morphizen/export.h>
#include <morphizen/morphizen-ort-api-ext.hpp>
/// header file used by ort MORPHIZEN execution providers.

using EventInfo = std::tuple<std::string, // name
                             int,         // pid
                             int,         // tid
                             long long,   // timestamp
                             long long    // duration
                             >;

namespace onnxruntime {
using ProviderOptions = std::unordered_map<std::string, std::string>;
}

// Forward declarations for logger types
namespace Ort {
struct Logger;
}

namespace morphizen {
class PassContextImp;
class LoggerAdapter;

using morphizen_error_report_func = void (*)(
    void*, int, const char*); // should be same as morphizen::error_report_func,
                              // for compile issue we  defined it

MORPHIZEN_DLL_SPEC void set_the_global_api(OrtApiForMorphizen* api);

MORPHIZEN_DLL_SPEC const OrtApiForMorphizen* api();

MORPHIZEN_DLL_SPEC std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_5(const std::filesystem::path& model_path,
                     const Graph& graph,
                     const onnxruntime::ProviderOptions& options, void* status,
                     morphizen_error_report_func func);

MORPHIZEN_DLL_SPEC std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_3(
    const std::string& model_path, const Graph& graph,
    const onnxruntime::ProviderOptions& options,
    const std::map<std::string, std::string>& session_configs,
    std::function<void(int, const char*)> set_ort_status = nullptr);

// Internal version that accepts logger_adapter for lifetime management
MORPHIZEN_DLL_SPEC std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_3_internal(
    const std::string& model_path, const Graph& onnx_graph,
    const onnxruntime::ProviderOptions& options,
    const std::map<std::string, std::string>& session_configs,
    std::unique_ptr<LoggerAdapter> logger_adapter = nullptr,
    std::function<void(int, const char*)> set_ort_status = nullptr);

MORPHIZEN_DLL_SPEC void profiler_collect(std::vector<EventInfo>& api_events,
                                         std::vector<EventInfo>& kernel_events);

int optimize_onnx_model(const std::filesystem::path& model_path_in,
                        const std::filesystem::path& model_path_out,
                        const char* json_config);

void initialize_graph_optimizer(const std::string& json_path);

} // namespace morphizen
