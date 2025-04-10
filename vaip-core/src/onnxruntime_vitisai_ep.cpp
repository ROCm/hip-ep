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
// #include "symbols.hpp"

// typedef void* void_ptr_t;
// #define DECLARE_SYMBOL(sym) extern "C" void_ptr_t sym;
// SYMBOLS(DECLARE_SYMBOL)
// #if defined(_WIN32)
// SYMBOLS_WIN(DECLARE_SYMBOL)
// #endif

// #define DEFINE_SYMBOL(sym) sym,

// static void_ptr_t reserved_symbols[] = {SYMBOLS(DEFINE_SYMBOL)};
#include "morphizen/onnxruntime_vitisai_ep.hpp"
#include "morphizen/config_reader.hpp"
#include "morphizen/op_def.hpp"
#include "morphizen/vaip.hpp"
#include <fstream>
#include <glog/logging.h>

namespace onnxruntime_vitisai_ep {
using namespace vaip_core;
int optimize_onnx_model(const std::filesystem::path& model_path_in,
                        const std::filesystem::path& model_path_out,
                        const char* json_config) {
  return vaip_core::optimize_onnx_model(model_path_in, model_path_out,
                                        json_config);
}
void initialize_graph_optimizer(const std::string& json_path) {
  vaip_core::initialize_graph_optimizer(json_path);
}
} // namespace onnxruntime_vitisai_ep

extern "C" {
VAIP_DLL_SPEC
const vaip_core::OrtApiForVaip* get_the_global_api() {
  // The test program is using this interface
  return vaip_core::api();
}

static void intialize_op_defs(std::vector<OrtCustomOpDomain*>& contrib_domains,
                              std::vector<OrtCustomOpDomain*>& ret_domain) {
  // This function is used to initialize the op_def_map
  typedef vaip_core::OpDefInfo* (*vaip_op_def_info_t)();
  auto op_def_info_ptrs =
      vaip_core::Plugin::get_all_symbols("vaip_op_def_info");
  for (auto op_def_into_ptr : op_def_info_ptrs) {
    auto op_def_info_func =
        reinterpret_cast<vaip_op_def_info_t>(op_def_into_ptr);
    auto op_def_info = op_def_info_func();
    std::vector<Ort::CustomOpDomain> domains;
    op_def_info->get_domains(domains);
    for (auto& domain : domains) {
      // Memory leak, passing data across dlls
      contrib_domains.push_back(domain.release());
      ret_domain.push_back(*contrib_domains.rbegin());
      CHECK_LE(ret_domain.size(), 100)
          << "ret_domain applied for 100 in onnxruntime";
    }
  }
    std::set<std::string> vitis_ep_custom_ops;
  // todo
  // for (const auto& domain : contrib_domains) {
  //  for (const auto* op : domain->custom_ops_) {
  //    vitis_ep_custom_ops.insert(domain->domain_ + "::" + op->GetName(op));
  //  }
  //}
  vitis_ep_custom_ops.insert("::DequantizeLinear");
  vitis_ep_custom_ops.insert("::QuantizeLinear");
  vitis_ep_custom_ops.insert("com.microsoft::DequantizeLinear");
  vitis_ep_custom_ops.insert("com.microsoft::QuantizeLinear");
  vaip_core::set_vitis_ep_custom_ops(vitis_ep_custom_ops);

}

// The interface exported below is used by onnxruntime_providers_vitisai.so
VAIP_DLL_SPEC
void initialize_onnxruntime_vitisai_ep(
    vaip_core::OrtApiForVaip* api,
    std::vector<OrtCustomOpDomain*>& ret_domain) {
  vaip_core::initialize_onnxruntime_vitisai_ep(api, ret_domain);
  static std::vector<OrtCustomOpDomain*> contrib_domains;
  intialize_op_defs(contrib_domains, ret_domain);
}
VAIP_DLL_SPEC
void deinitialize_onnxruntime_vitisai_ep() {
  vaip_core::deinitialize_onnxruntime_vitisai_ep();
}

VAIP_DLL_SPEC
std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>*
compile_onnx_model_vitisai_ep_with_options(
    const std::string& model_path, const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options) {
  auto json_config = vaip_core::get_config_json_str(options);
  return new std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>(
      vaip_core::compile_onnx_model_3(model_path, graph, json_config.c_str()));
}

VAIP_DLL_SPEC
std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>*
compile_onnx_model_vitisai_ep_with_error_handling(
    const std::string& model_path, const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options, void* status,
    void (*func)(void*, int, const char*)) {
  auto json_config = vaip_core::get_config_json_str(options);
  return new std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>(
      vaip_core::compile_onnx_model_3(model_path, graph, json_config.c_str()));
}

VAIP_DLL_SPEC
void profiler_collect(std::vector<EventInfo>& api_events,
                      std::vector<EventInfo>& kernel_events) {
  vaip_core::profiler_collect(api_events, kernel_events);
}
}
