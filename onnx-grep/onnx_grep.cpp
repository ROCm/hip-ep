/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// clang-format off
#include <glog/logging.h>
#include <exception>
#include <limits>
#define ORT_API_MANUAL_INIT 1
#include "onnxruntime_cxx_api.h"
#include "morphizen/vaip.hpp"

extern "C" {
#include "./getopt.h"
}

#ifdef CREATE_DUMMY_SESSION
#include <codecvt>
#include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#endif// CREATE_DUMMY_SESSION

static bool endsWith(const std::string &fullString, const std::string &ending) {
  if (fullString.size() >= ending.size()) {
      return (fullString.compare(fullString.length() - ending.size(), ending.size(), ending) == 0);
  } else {
      return false;
  }
}

static std::shared_ptr<vaip_core::Pattern> get_pattern(const std::string& file) {
    auto builder = vaip_core::PatternBuilder ();
    auto ret = std::shared_ptr<vaip_core::Pattern>();
  // see test_conv_pattern.py as an example
  if (endsWith(file, std::string(".py"))) {
#ifdef ENABLE_PYTHON
      ret =  builder.create_by_py(vaip_core::slurp(file));
#else
    throw std::runtime_error("Unsupported pattern data type");
#endif
  } else if (endsWith(file, std::string(".json"))) {
      ret =  builder.create_by_json(vaip_core::slurp(file));
  } else {
      LOG(ERROR) << "cannot pattern " << file << ", pattern file only support json and python file";
  }
  return ret;
}

static void usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " -f <onnx_model> -p <pattern_file> [-n <node_arg>] [-v]"
            << std::endl;
  std::cout << "    -f <onnx_model> : onnx model file" << std::endl;
  std::cout
      << "    -p <pattern_file> : pattern file, can be json or python"
      << std::endl;
  std::cout << "    -n <node_arg> : node arg name to trace" << std::endl;
  std::cout << "    -v : verbose mode" << std::endl;
  std::cout << "    -h : help" << std::endl;
}

int main(int argc, char* argv[]) {
  Ort::InitApi();
  std::cout << "- ONNX Grep utility ..." << std::endl;
  try {
    auto file = std::string();
    auto pattern = std::string();
    auto node_arg = std::string();
    auto opt_verbose = false;
    int opt = 0;
    while ((opt = getopt(argc, argv, "p:f:n:vh")) != -1) {
      switch (opt) {
      case 'f': {
        file = std::string(optarg);
        break;
      }
      case 'p': {
        pattern = std::string(optarg);
        break;
      }
      case 'n': {
        node_arg = std::string(optarg);
        break;
      }
      case 'v': {
        opt_verbose = true;
        break;
      }
      case 'h': {
        usage(argv[0]);
        exit(0);
      }
      }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "onnx_grep");
    Ort::SessionOptions().AppendExecutionProvider_VitisAI();
         vaip_core::set_the_global_api(
          vaip_core::Plugin::invoke<vaip_core::OrtApiForVaip*>(
              "onnxruntime_vitisai_ep", "get_the_global_api"));
    CHECK_NE(file, "");

    auto p = get_pattern(pattern);
    if (p == nullptr) {
      LOG(ERROR) << "no pattern";
      return 1;
    }
    if (opt_verbose) {
      std::cout << "pattern is " << (void*)p.get() << std::endl;
      std::cout << "pattern is " << p->debug_string() << std::endl;
    }
    if (!node_arg.empty()) {
      vaip_core::Pattern::enable_trace(1);
    }
    auto model = vaip_core::model_load(std::filesystem::path(file).u8string());
    auto& graph = VAIP_ORT_API(model_main_graph)(*model);
    vaip_core::graph_resolve(graph, true);
    if (!node_arg.empty()) {
      auto node_found = VAIP_ORT_API(graph_producer_node)(graph, node_arg);
      CHECK(node_found != nullptr)
          << "cannot find node arg. node_arg=" << node_arg;
    }
    for (auto index : vaip_core::graph_get_node_in_topoligical_order(graph)) {
      auto node = VAIP_ORT_API(graph_get_node)(graph, index);
      CHECK(node != nullptr);
      auto this_node_arg_name = vaip_core::node_get_first_output_name(*node);
      // node_arg.empty() means user does not specify `-n` for
      // tracing, we try to search for all possible matched node.
      //
      // if it is not empty, we only trace the node whose name is
      // `node_arg`, i.e. this_node_arg_name == node_arg.
      if (node_arg.empty() || (this_node_arg_name == node_arg)) {
        auto bind = p->match(graph, *node);
        if (bind != nullptr) {
          LOG(INFO) << "find node: " << vaip_core::node_as_string(*node);
          if (opt_verbose) {
            for (auto ni : *bind) {
              LOG(INFO) << "pattern id: " << ni.first << " node_arg: "
                        << vaip_core::node_arg_as_string(*ni.second.node_arg);
            }
          }
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }

  return 0;
}

#include "./getopt.c"
