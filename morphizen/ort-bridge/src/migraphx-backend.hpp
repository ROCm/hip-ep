/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
// EP-side handle to a whole graph compiled by the MIGraphX backend plugin
// (lib/Backend/MIGraphX, ABI in migraphx-plugin-abi.hpp).
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct hipep_mgx_program;

namespace morphizen {

struct MigraphxPlugin;

class MigraphxProgram {
public:
  // Loads the plugin from the EP library's directory on first use. Returns
  // nullptr and sets `error` when the plugin is unavailable or the compile
  // fails; never throws.
  static std::shared_ptr<MigraphxProgram>
  compile(const std::string &model_path,
          const std::vector<std::string> &input_names,
          const std::vector<std::vector<int64_t>> &input_dims,
          std::string &error);

  ~MigraphxProgram();
  MigraphxProgram(const MigraphxProgram &) = delete;
  MigraphxProgram &operator=(const MigraphxProgram &) = delete;

  const std::string &model_path() const { return model_path_; }
  const std::vector<std::string> &input_names() const { return input_names_; }
  const std::vector<std::vector<int64_t>> &input_dims() const {
    return input_dims_;
  }
  const std::vector<std::vector<int64_t>> &output_dims() const {
    return output_dims_;
  }

  // Inputs in input_names() order, outputs in graph-output order; all are
  // device buffers. Returns after the GPU work completes. Throws
  // std::runtime_error on failure.
  void run(const std::vector<void *> &inputs,
           const std::vector<void *> &outputs);

private:
  MigraphxProgram() = default;

  const MigraphxPlugin *plugin_ = nullptr;
  hipep_mgx_program *program_ = nullptr;
  std::string model_path_;
  std::vector<std::string> input_names_;
  std::vector<std::vector<int64_t>> input_dims_;
  std::vector<std::vector<int64_t>> output_dims_;
  // A compiled MIGraphX program is not safe for concurrent eval.
  std::mutex run_mutex_;
};

} // namespace morphizen
