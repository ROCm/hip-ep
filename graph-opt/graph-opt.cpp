/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// clang-format off
#include <exception>
#include <filesystem>
#include <limits>
#include <glog/logging.h>
#include <boost/program_options.hpp>
#include "morphizen/vaip.hpp"
// clang-format on

namespace po = boost::program_options;
using namespace morphizen;
using namespace std;

// Validate file path to mitigate path traversal risks
static bool validate_path(const std::string& path,
                          const std::string& param_name) {
  if (path.empty()) {
    return true; // Empty paths are handled by subsequent checks
  }

  // Check for null bytes (potential for path truncation attacks)
  if (path.find('\0') != std::string::npos) {
    std::cerr << "Error: " << param_name << " contains null byte: " << path
              << std::endl;
    return false;
  }

  // Warn about suspicious patterns but don't block them (users may legitimately
  // need them)
  if (path.find("..") != std::string::npos) {
    std::cerr << "Warning: " << param_name << " contains '..': " << path
              << std::endl;
  }

  return true;
}

int main(int argc, char* argv[]) {
  try {
    // Define command line options
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")(
        "input,i", po::value<std::string>(), "input onnx model file")(
        "output,o", po::value<std::string>(), "output onnx model file")(
        "output-txt,t", po::value<std::string>(), "output model to txt file")(
        "pass,p", po::value<std::vector<std::string>>()->multitoken(),
        "pass list (can be specified multiple times)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Handle help option
    if (vm.count("help")) {
      std::cout << "Usage: " << argv[0] << " [options]\n" << desc << std::endl;
      return 0;
    }

    // Extract option values
    auto opt_input_file =
        vm.count("input") ? vm["input"].as<std::string>() : std::string();
    auto opt_output_file =
        vm.count("output") ? vm["output"].as<std::string>() : std::string();
    auto opt_output_txt_file = vm.count("output-txt")
                                   ? vm["output-txt"].as<std::string>()
                                   : std::string();
    auto opt_pass = vm.count("pass") ? vm["pass"].as<std::vector<std::string>>()
                                     : std::vector<std::string>();

    // Validate all paths at entry point
    if (!validate_path(opt_input_file, "input")) {
      return 1;
    }
    if (!validate_path(opt_output_file, "output")) {
      return 1;
    }
    if (!validate_path(opt_output_txt_file, "output-txt")) {
      return 1;
    }

    std::shared_ptr<PassContext> context = PassContext::create();

    auto protos = std::vector<std::unique_ptr<PassProto>>{};
    auto passes = std::vector<std::shared_ptr<morphizen::IPass>>{};
    for (auto& opt_pass_i : opt_pass) {
      protos.emplace_back(std::make_unique<PassProto>());
      auto& pass_proto = *protos.back();
      pass_proto.set_name("test");
      pass_proto.set_plugin(opt_pass_i);
      passes.emplace_back(IPass::create_pass(context, pass_proto));
    }

    auto model = morphizen::model_load(opt_input_file);
    auto model_ref = morphizen_cxx::ModelConstRef(*model);
    auto graph_ref = model_ref.main_graph();
    auto& graph = graph_ref;
    graph_resolve(graph);

    IPass::run_passes(passes, graph);

    if (!opt_output_file.empty()) {
      LOG(INFO) << "write output file to " << opt_output_file;
      graph_ref.save(opt_output_file, opt_output_file + ".dat",
                     (std::numeric_limits<size_t>::max)());
    }
    if (!opt_output_txt_file.empty()) {
      LOG(INFO) << "write output file to " << opt_output_txt_file;
      morphizen::dump_graph(graph, opt_output_txt_file);
    }
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }

  return 0;
}
