/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <exception>
#include <filesystem>
#include <glog/logging.h>

#include "initialize_vaip.hpp"
#include <limits>

//
#include "morphizen/vaip.hpp"
//

extern "C" {
#include "getopt.h"
}

using namespace vaip_core;
using namespace std;
int main(int argc, char* argv[]) {
  try {
    auto opt_input_file = std::string();
    auto opt_cache = std::string();
    auto opt_output_file = std::string();
    auto opt_output_txt_file = std::string();
    auto opt_module_name = std::string();
    auto opt_method_name = std::string("rules");
    auto opt_pass = std::vector<std::string>();
    int opt = 0;
    while ((opt = getopt(argc, argv, "m:i:f:o:t:c:p:")) != -1) {
      switch (opt) {
      case 'i': {
        opt_input_file = std::string(optarg);
        break;
      }
      case 'o': {
        opt_output_file = std::string(optarg);
        break;
      }
      case 't': {
        opt_output_txt_file = std::string(optarg);
        break;
      }
      case 'm': {
        opt_module_name = std::string(optarg);
        break;
      }
      case 'f': {
        opt_method_name = std::string(optarg);
        break;
      }
      case 'p': {
        opt_pass.push_back(std::string(optarg));
        break;
      }
      case 'c': {
        opt_cache = std::string(optarg);
        break;
      }
      }
    }
    try {
      Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "morphizen-graph-opt");
      Ort::SessionOptions().AppendExecutionProvider_VitisAI();
      vaip_core::set_the_global_api(
          vaip_core::Plugin::invoke<vaip_core::OrtApiForVaip*>(
              "onnxruntime_vitisai_ep", "get_the_global_api"));
    } catch (const std::exception& e) {
      std::cerr << "exception occurs : " << e.what() << "\n";
    }
    std::shared_ptr<PassContext> context = PassContext::create();
    if (!opt_cache.empty()) {
      context = load_context(opt_cache);
    }
    auto protos = std::vector<std::unique_ptr<PassProto>>{};
    auto passes = std::vector<std::shared_ptr<vaip_core::IPass>>{};
    for (auto& opt_pass_i : opt_pass) {
      protos.emplace_back(std::make_unique<PassProto>());
      auto& pass_proto = *protos.back();
      pass_proto.set_name("test");
      pass_proto.set_plugin(opt_pass_i);
      passes.emplace_back(IPass::create_pass(context, pass_proto));
    }

    auto model = vaip_core::model_load(opt_input_file);
    auto& graph = VAIP_ORT_API(model_main_graph)(*model);
    graph_resolve(graph);

    IPass::run_passes(passes, graph);

    if (!opt_output_file.empty()) {
      LOG(INFO) << "write output file to " << opt_output_file;
      VAIP_ORT_API(graph_save)
      (graph, opt_output_file, opt_output_file + ".dat",
       std::numeric_limits<size_t>::max());
    }
    if (!opt_output_txt_file.empty()) {
      LOG(INFO) << "write output file to " << opt_output_txt_file;
      vaip_core::dump_graph(graph, opt_output_txt_file);
    }
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
  return 0;
}

#include "getopt.c"
