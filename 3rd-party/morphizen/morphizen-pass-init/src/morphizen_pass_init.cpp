/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
using namespace morphizen;
DEF_ENV_PARAM(XLNX_ENABLE_DUMP_ONNX_MODEL, "0")

struct InitPass {
  InitPass(IPass & /*self*/) {}
  void process(IPass &self, Graph &graph) {
    if (ENV_PARAM(XLNX_ENABLE_DUMP_ONNX_MODEL) ||
        self.get_context()->get_provider_option("pass.init.enable_dump", "0") ==
            "1") {
      auto default_dump_dir =
          self.get_context()->get_dump_directory().u8string();
      auto log_dir = self.get_context()->get_provider_option(
          "pass.init.directory", default_dump_dir);
      auto log_dir_path = std::filesystem::u8path(log_dir);
      if (!std::filesystem::exists(log_dir_path)) {
        std::filesystem::create_directories(log_dir_path);
      }
      auto default_onnx_file_name = "onnx.onnx";
      auto onnx_file_name = self.get_context()->get_provider_option(
          "pass.init.filename", default_onnx_file_name);
      auto onnx_file_path =
          log_dir_path / std::filesystem::u8path(onnx_file_name);
      auto onnx_file_data_path = log_dir_path / "onnx.dat";
      morphizen_cxx::GraphRef(graph).set_name("resent50_by_morphizen");

      morphizen_cxx::GraphConstRef(graph).save(
          onnx_file_path, onnx_file_data_path,
          std::numeric_limits<size_t>::max());
      LOG(INFO) << "save origin onnx model to " << onnx_file_name << " data in "
                << onnx_file_data_path;
    }
    return;
  }
};

DEFINE_MORPHIZEN_PASS(InitPass, morphizen_pass_init)
