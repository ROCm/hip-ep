/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Fake MIGraphX backend plugin for GPU-free tests of the EP's MIGraphX path
// (routing, claim, compile, compute and error handling). It implements
// migraphx-plugin-abi.hpp without MIGraphX:
//
// - compile: fails when the model file is missing or MOCK_MGX_FAIL=compile.
//   Reports MOCK_MGX_NUM_OUTPUTS outputs (default 1), each shaped like
//   input 0.
// - run: fails when MOCK_MGX_FAIL=run. Otherwise, for float tensors,
//   output[j][k] = sum_i (i + 1) * input[i][k] + 100 * (j + 1)
//   so a test can tell which program input and output each buffer was bound
//   to, and that this plugin (not HIP or the CPU EP) produced the result.
#include "migraphx-plugin-abi.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#define MOCK_MGX_EXPORT __declspec(dllexport)
#else
#define MOCK_MGX_EXPORT __attribute__((visibility("default")))
#endif

struct hipep_mgx_program {
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  std::vector<size_t> dims;
  size_t elements = 1;
};

namespace {

bool fail_requested(const char *stage) {
  const char *v = std::getenv("MOCK_MGX_FAIL");
  return v != nullptr && std::strcmp(v, stage) == 0;
}

int fail(char *err, size_t err_size, const std::string &msg) {
  if (err != nullptr && err_size > 0) {
    std::strncpy(err, msg.c_str(), err_size - 1);
    err[err_size - 1] = '\0';
  }
  return 1;
}

} // namespace

extern "C" {

MOCK_MGX_EXPORT int hipep_mgx_abi_version(void) {
  return HIPEP_MGX_PLUGIN_ABI_VERSION;
}

MOCK_MGX_EXPORT int hipep_mgx_compile(const char *model_path, size_t num_inputs,
                                      const char *const * /*input_names*/,
                                      const size_t *const *input_dims,
                                      const size_t *input_ranks,
                                      hipep_mgx_program **out_program,
                                      char *err, size_t err_size) {
  *out_program = nullptr;
  if (fail_requested("compile"))
    return fail(err, err_size, "mock compile failure");
  if (!std::filesystem::exists(std::filesystem::u8path(model_path)))
    return fail(err, err_size, std::string("mock: no model at ") + model_path);
  if (num_inputs == 0)
    return fail(err, err_size, "mock: graph has no inputs");
  auto *p = new hipep_mgx_program;
  p->num_inputs = num_inputs;
  const char *n = std::getenv("MOCK_MGX_NUM_OUTPUTS");
  p->num_outputs = n != nullptr ? std::strtoul(n, nullptr, 10) : 1;
  p->dims.assign(input_dims[0], input_dims[0] + input_ranks[0]);
  for (size_t d : p->dims)
    p->elements *= d;
  *out_program = p;
  return 0;
}

MOCK_MGX_EXPORT size_t hipep_mgx_num_outputs(const hipep_mgx_program *p) {
  return p->num_outputs;
}

MOCK_MGX_EXPORT int hipep_mgx_output_shape(const hipep_mgx_program *p,
                                           size_t index, size_t *dims,
                                           size_t max_rank, size_t *rank) {
  if (index >= p->num_outputs || p->dims.size() > max_rank)
    return 1;
  std::memcpy(dims, p->dims.data(), p->dims.size() * sizeof(size_t));
  *rank = p->dims.size();
  return 0;
}

MOCK_MGX_EXPORT int hipep_mgx_run(hipep_mgx_program *p, size_t num_inputs,
                                  const char *const * /*input_names*/,
                                  void *const *input_data, size_t num_outputs,
                                  void *const *output_data, char *err,
                                  size_t err_size) {
  if (fail_requested("run"))
    return fail(err, err_size, "mock run failure");
  if (num_inputs != p->num_inputs || num_outputs != p->num_outputs)
    return fail(err, err_size, "mock: input/output count mismatch");
  for (size_t j = 0; j < num_outputs; ++j) {
    auto *out = static_cast<float *>(output_data[j]);
    for (size_t k = 0; k < p->elements; ++k) {
      float v = 100.0f * static_cast<float>(j + 1);
      for (size_t i = 0; i < num_inputs; ++i)
        v += static_cast<float>(i + 1) *
             static_cast<const float *>(input_data[i])[k];
      out[k] = v;
    }
  }
  return 0;
}

MOCK_MGX_EXPORT void hipep_mgx_release(hipep_mgx_program *p) { delete p; }

} // extern "C"
