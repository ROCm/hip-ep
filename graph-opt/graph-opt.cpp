/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// clang-format off
#include <exception>
#include <filesystem>
#include <limits>
#include <glog/logging.h>
#include "onnxruntime_cxx_api.h"
#include "morphizen/vaip.hpp"
// clang-format on

extern "C" {
#include "getopt.h"
}

using namespace vaip_core;
using namespace std;

static void usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " -i <input_onnx_model> -o <output_onnx_model> -t "
               "<output_txt_file> -p <vaip-pass_1> [vaip-pass_2 ...] [-h]"
            << std::endl;
  std::cout << "\t-i <input_onnx_model> : input onnx model file" << std::endl;
  std::cout << "\t-o <output_onnx_model> : output onnx model file" << std::endl;
  std::cout << "\t-t <output_txt_file> : output model to txt file" << std::endl;
  std::cout << "\t-p <vaip-pass_1> [vaip-pass_2 ...] : pass list" << std::endl;
  std::cout << "\t-h : help" << std::endl;
  return;
}

int main(int argc, char* argv[]) {
  try {
    auto opt_input_file = std::string();
    auto opt_cache = std::string();
    auto opt_output_file = std::string();
    auto opt_output_txt_file = std::string();
    auto opt_pass = std::vector<std::string>();
    int opt = 0;
    while ((opt = getopt(argc, argv, "i:o:t:c:p:h")) != -1) {
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
      case 'p': {
        opt_pass.push_back(std::string(optarg));
        break;
      }
      case 'c': {
        opt_cache = std::string(optarg);
        break;
      }
      case 'h': {
        usage(argv[0]);
        exit(0);
      }
      }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "morphizen-graph-opt");
    try {
      Ort::SessionOptions().AppendExecutionProvider_VitisAI();
    } catch (const std::exception& e) {
      std::cerr << "exception occurs : " << e.what() << "\n";
      return 1;
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
       (std::numeric_limits<size_t>::max)());
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
