/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <glog/logging.h>

#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
using namespace vaip_core;
DEF_ENV_PARAM(XLNX_ENABLE_DUMP_ONNX_MODEL, "0")

struct InitPass {
  InitPass(IPass& self) {}
  void process(IPass& self, Graph& graph) {
    if (ENV_PARAM(XLNX_ENABLE_DUMP_ONNX_MODEL)) {
      auto log_dir = self.get_log_path();
      if (log_dir.empty()) {
        LOG(WARNING) << "log dir is empty, call saving onnx.onnx";
        return;
      }
      auto file = log_dir / "onnx.onnx";
      auto dat_file = "onnx.dat";
      vaip_cxx::GraphConstRef(graph).save(file.u8string(), dat_file,
                                          std::numeric_limits<size_t>::max());
      LOG(INFO) << "save origin onnx model to " << file << " data in "
                << dat_file;
    }
    return;
  }
};

DEFINE_VAIP_PASS(InitPass, vaip_pass_init)
