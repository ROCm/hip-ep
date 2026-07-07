/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// clang-format off
#include <glog/logging.h>
#include <exception>
#include <limits>
#include <boost/program_options.hpp>
#include "morphizen/vaip.hpp"

namespace po = boost::program_options;

// Validate file path to mitigate path traversal risks
static bool validate_path(const std::string& path, const std::string& param_name) {
  if (path.empty()) {
    return true;  // Empty paths are handled by subsequent checks
  }

  // Check for null bytes (potential for path truncation attacks)
  if (path.find('\0') != std::string::npos) {
    std::cerr << "Error: " << param_name << " contains null byte: " << path << std::endl;
    return false;
  }

  // Warn about suspicious patterns but don't block them (users may legitimately need them)
  if (path.find("..") != std::string::npos) {
    std::cerr << "Warning: " << param_name << " contains '..': " << path << std::endl;
  }

  return true;
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

static std::shared_ptr<morphizen::Pattern> get_pattern(const std::string& file) {
    auto builder = morphizen::PatternBuilder ();
    auto ret = std::shared_ptr<morphizen::Pattern>();
  // see test_conv_pattern.py as an example
  if (endsWith(file, std::string(".py"))) {
#ifdef ENABLE_PYTHON
      ret =  builder.create_by_py(morphizen::slurp(file));
#else
    throw std::runtime_error("Unsupported pattern data type");
#endif
  } else if (endsWith(file, std::string(".json"))) {
      ret =  builder.create_by_json(morphizen::slurp(file));
  } else {
      LOG(ERROR) << "cannot pattern " << file << ", pattern file only support json and python file";
  }
  return ret;
}

int main(int argc, char* argv[]) {
  std::cout << "- ONNX Grep utility ..." << std::endl;
  try {
    // Define command line options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("file,f", po::value<std::string>(), "onnx model file")
        ("pattern,p", po::value<std::string>(), "pattern file (can be json or python)")
        ("node-arg,n", po::value<std::string>(), "node arg name to trace")
        ("verbose,v", "verbose mode");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Handle help option
    if (vm.count("help")) {
      std::cout << "Usage: " << argv[0] << " [options]\n" << desc << std::endl;
      return 0;
    }

    // Extract option values
    auto file = vm.count("file") ? vm["file"].as<std::string>() : std::string();
    auto pattern = vm.count("pattern") ? vm["pattern"].as<std::string>() : std::string();
    auto node_arg = vm.count("node-arg") ? vm["node-arg"].as<std::string>() : std::string();
    auto opt_verbose = vm.count("verbose") > 0;

    // Validate paths at entry point
    if (!validate_path(file, "file")) {
      return 1;
    }
    if (!validate_path(pattern, "pattern")) {
      return 1;
    }

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
      morphizen::Pattern::enable_trace(1);
    }
    auto model = morphizen::model_load(std::filesystem::path(file).u8string());
    auto model_ref = morphizen_cxx::ModelConstRef(*model);
    auto graph_ref = model_ref.main_graph();
    auto& graph = graph_ref;
    morphizen::graph_resolve(graph, true);
    if (!node_arg.empty()) {
      auto node_arg_opt = graph_ref.find_node_arg(node_arg);
      CHECK(node_arg_opt.has_value())
          << "cannot find node arg. node_arg=" << node_arg;
      auto node_found = node_arg_opt.value().find_producer();
      CHECK(node_found.has_value())
          << "cannot find producer node for node_arg=" << node_arg;
    }
    for (auto index : morphizen::graph_get_node_in_topoligical_order(graph)) {
      auto node_opt = graph_ref.find_node(index);
      CHECK(node_opt.has_value());
      auto node = node_opt.value().ptr();
      auto node_ref = node_opt.value();
      auto this_node_arg_name = morphizen::node_arg_get_name(node_ref.first_output_node_arg());
      // node_arg.empty() means user does not specify `-n` for
      // tracing, we try to search for all possible matched node.
      //
      // if it is not empty, we only trace the node whose name is
      // `node_arg`, i.e. this_node_arg_name == node_arg.
      if (node_arg.empty() || (this_node_arg_name == node_arg)) {
        auto bind = p->match(graph, *node);
        if (bind != nullptr) {
          LOG(INFO) << "find node: " << node_ref.to_string();
          if (opt_verbose) {
            for (auto ni : *bind) {
              auto node_arg_ref = morphizen_cxx::NodeArgConstRef::from_node_arg(graph, *ni.second.node_arg);
              LOG(INFO) << "pattern id: " << ni.first << " node_arg: "
                        << node_arg_ref.to_string();
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
