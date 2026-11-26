/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/morphizen.hpp"

#include "./migraphx-backend.hpp"
#include "./migraphx-plugin-abi.hpp"
#include "morphizen-foundation/env_config.hpp"
#include "morphizen-utils/morphizen_plugin.hpp"

#include <filesystem>
#include <stdexcept>

// Directory holding the plugin; empty means the EP library's directory.
DEF_ENV_PARAM_2(HIPDNN_EP_MIGRAPHX_PLUGIN_DIR, "", std::string)

namespace morphizen {

struct MigraphxPlugin {
  hipep_mgx_compile_fn compile = nullptr;
  hipep_mgx_num_outputs_fn num_outputs = nullptr;
  hipep_mgx_output_shape_fn output_shape = nullptr;
  hipep_mgx_run_fn run = nullptr;
  hipep_mgx_release_fn release = nullptr;
};

namespace {

constexpr size_t kErrSize = 1024;
constexpr size_t kMaxRank = 16;

// The plugin sits next to the EP library: hipep_migraphx.so on Linux (the
// loader adds no "lib" prefix to a path), hipep_migraphx.dll on Windows.
const MigraphxPlugin *load_plugin(std::string &error) {
  static MigraphxPlugin plugin;
  static std::string load_error;
  static std::once_flag once;
  std::call_once(once, [] {
    const std::string &dir = ENV_PARAM(HIPDNN_EP_MIGRAPHX_PLUGIN_DIR);
    const auto path = ((dir.empty() ? get_morphizen_path().parent_path()
                                    : std::filesystem::u8path(dir)) /
                       "hipep_migraphx")
                          .u8string();
    auto *handle = Plugin::get(std::string(path.begin(), path.end()));
    if (handle == nullptr) {
      load_error = "MIGraphX backend plugin not found or not loadable at " +
                   std::string(path.begin(), path.end());
      return;
    }
    auto version = handle->get_method<int>("hipep_mgx_abi_version");
    if (version == nullptr || version() != HIPEP_MGX_PLUGIN_ABI_VERSION) {
      load_error = "MIGraphX backend plugin ABI version mismatch";
      return;
    }
    plugin.compile =
        handle->get_method<int, const char *, size_t, const char *const *,
                           const size_t *const *, const size_t *,
                           hipep_mgx_program **, char *, size_t>(
            "hipep_mgx_compile");
    plugin.num_outputs = handle->get_method<size_t, const hipep_mgx_program *>(
        "hipep_mgx_num_outputs");
    plugin.output_shape =
        handle->get_method<int, const hipep_mgx_program *, size_t, size_t *,
                           size_t, size_t *>("hipep_mgx_output_shape");
    plugin.run =
        handle
            ->get_method<int, hipep_mgx_program *, size_t, const char *const *,
                         void *const *, size_t, void *const *, char *, size_t>(
                "hipep_mgx_run");
    plugin.release =
        handle->get_method<void, hipep_mgx_program *>("hipep_mgx_release");
    if (!plugin.compile || !plugin.num_outputs || !plugin.output_shape ||
        !plugin.run || !plugin.release) {
      plugin = MigraphxPlugin{};
      load_error = "MIGraphX backend plugin is missing an entry point";
    }
  });
  if (plugin.compile == nullptr) {
    error = load_error;
    return nullptr;
  }
  return &plugin;
}

} // namespace

std::shared_ptr<MigraphxProgram> MigraphxProgram::compile(
    const std::string &model_path, const std::vector<std::string> &input_names,
    const std::vector<std::vector<int64_t>> &input_dims, std::string &error) {
  const MigraphxPlugin *plugin = load_plugin(error);
  if (plugin == nullptr)
    return nullptr;

  std::vector<const char *> names;
  std::vector<std::vector<size_t>> dims;
  std::vector<const size_t *> dim_ptrs;
  std::vector<size_t> ranks;
  for (size_t i = 0; i < input_names.size(); ++i) {
    names.push_back(input_names[i].c_str());
    dims.emplace_back(input_dims[i].begin(), input_dims[i].end());
  }
  for (const auto &d : dims) {
    dim_ptrs.push_back(d.data());
    ranks.push_back(d.size());
  }

  hipep_mgx_program *raw = nullptr;
  char err[kErrSize] = {};
  if (plugin->compile(model_path.c_str(), names.size(), names.data(),
                      dim_ptrs.data(), ranks.data(), &raw, err,
                      sizeof(err)) != 0) {
    error = err;
    return nullptr;
  }

  std::shared_ptr<MigraphxProgram> result(new MigraphxProgram());
  result->plugin_ = plugin;
  result->program_ = raw;
  result->model_path_ = model_path;
  result->input_names_ = input_names;
  result->input_dims_ = input_dims;
  const size_t num_outputs = plugin->num_outputs(raw);
  for (size_t i = 0; i < num_outputs; ++i) {
    size_t shape[kMaxRank];
    size_t rank = 0;
    if (plugin->output_shape(raw, i, shape, kMaxRank, &rank) != 0) {
      error = "MIGraphX output " + std::to_string(i) + " has no static shape";
      return nullptr;
    }
    result->output_dims_.emplace_back(shape, shape + rank);
  }
  return result;
}

MigraphxProgram::~MigraphxProgram() {
  if (program_ != nullptr)
    plugin_->release(program_);
}

void MigraphxProgram::run(const std::vector<void *> &inputs,
                          const std::vector<void *> &outputs) {
  std::vector<const char *> names;
  for (const auto &name : input_names_)
    names.push_back(name.c_str());
  char err[kErrSize] = {};
  std::lock_guard<std::mutex> lock(run_mutex_);
  if (plugin_->run(program_, names.size(), names.data(), inputs.data(),
                   outputs.size(), outputs.data(), err, sizeof(err)) != 0) {
    throw std::runtime_error(err);
  }
}

} // namespace morphizen
