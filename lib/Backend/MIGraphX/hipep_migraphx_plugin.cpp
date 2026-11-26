/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// MIGraphX backend plugin: implements migraphx-plugin-abi.hpp on top of the
// MIGraphX C++ API (a header-only wrapper over the stable migraphx::c library).
// Only this library links MIGraphX; the EP loads it at run time.
#include "migraphx-plugin-abi.hpp"

#include <migraphx/migraphx.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define HIPEP_MGX_EXPORT __declspec(dllexport)
#else
#define HIPEP_MGX_EXPORT __attribute__((visibility("default")))
#endif

struct hipep_mgx_program {
  migraphx::program program;
  // With offload_copy=false every graph output becomes a program parameter
  // named "<module>:#output_<i>"; index i is the graph output index.
  std::vector<std::string> output_params;
  std::vector<migraphx::shape> output_shapes;
  migraphx::program_parameter_shapes param_shapes;
};

namespace {

void set_error(char *err, size_t err_size, const std::string &msg) {
  if (err == nullptr || err_size == 0)
    return;
  const size_t n = std::min(msg.size(), err_size - 1);
  std::memcpy(err, msg.data(), n);
  err[n] = '\0';
}

// The MIGraphX C++ wrapper throws std::runtime_error("Failed to call
// function") and prints the underlying reason to stderr.
std::string describe_current_exception(const char *what_failed) {
  try {
    throw;
  } catch (const std::exception &e) {
    return std::string(what_failed) + ": " + e.what() +
           " (MIGraphX prints details to stderr)";
  } catch (...) {
    return std::string(what_failed) + ": unknown exception";
  }
}

// Returns the graph output index encoded in an output parameter name, or -1.
long output_index(const std::string &name) {
  static const std::string marker = "#output_";
  const auto pos = name.rfind(marker);
  if (pos == std::string::npos)
    return -1;
  const std::string digits = name.substr(pos + marker.size());
  if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](char c) {
        return c >= '0' && c <= '9';
      }))
    return -1;
  return std::stol(digits);
}

} // namespace

extern "C" {

HIPEP_MGX_EXPORT int hipep_mgx_abi_version(void) {
  return HIPEP_MGX_PLUGIN_ABI_VERSION;
}

HIPEP_MGX_EXPORT int
hipep_mgx_compile(const char *model_path, size_t num_inputs,
                  const char *const *input_names,
                  const size_t *const *input_dims, const size_t *input_ranks,
                  hipep_mgx_program **out_program, char *err, size_t err_size) {
  *out_program = nullptr;
  try {
    migraphx::onnx_options onnx_options;
    for (size_t i = 0; i < num_inputs; ++i) {
      onnx_options.set_input_parameter_shape(
          input_names[i],
          std::vector<size_t>(input_dims[i], input_dims[i] + input_ranks[i]));
    }
    auto program = migraphx::parse_onnx(model_path, onnx_options);

    migraphx::compile_options compile_options;
    // Bind ORT's device buffers directly; no host round trip.
    compile_options.set_offload_copy(false);
    // Match the numerics of ORT's MIGraphX EP.
    compile_options.set_fast_math(false);
    program.compile(migraphx::target("gpu"), compile_options);

    auto result = new hipep_mgx_program{std::move(program), {}, {}, {}};
    result->param_shapes = result->program.get_parameter_shapes();
    std::vector<std::pair<long, std::string>> outputs;
    for (const char *name : result->param_shapes.names()) {
      const long index = output_index(name);
      if (index >= 0)
        outputs.emplace_back(index, name);
    }
    std::sort(outputs.begin(), outputs.end());
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].first != static_cast<long>(i)) {
        delete result;
        set_error(err, err_size,
                  "MIGraphX output parameters are not numbered 0..N-1");
        return 1;
      }
      result->output_params.push_back(outputs[i].second);
      result->output_shapes.push_back(
          result->param_shapes[outputs[i].second.c_str()]);
    }
    *out_program = result;
    return 0;
  } catch (...) {
    set_error(err, err_size, describe_current_exception("MIGraphX compile"));
    return 1;
  }
}

HIPEP_MGX_EXPORT size_t
hipep_mgx_num_outputs(const hipep_mgx_program *program) {
  return program->output_params.size();
}

HIPEP_MGX_EXPORT int hipep_mgx_output_shape(const hipep_mgx_program *program,
                                            size_t index, size_t *dims,
                                            size_t max_rank, size_t *rank) {
  if (index >= program->output_shapes.size())
    return 1;
  const auto lengths = program->output_shapes[index].lengths();
  if (lengths.size() > max_rank)
    return 1;
  std::copy(lengths.begin(), lengths.end(), dims);
  *rank = lengths.size();
  return 0;
}

HIPEP_MGX_EXPORT int hipep_mgx_run(hipep_mgx_program *program,
                                   size_t num_inputs,
                                   const char *const *input_names,
                                   void *const *input_data, size_t num_outputs,
                                   void *const *output_data, char *err,
                                   size_t err_size) {
  try {
    if (num_outputs != program->output_params.size()) {
      set_error(err, err_size, "MIGraphX run: output count mismatch");
      return 1;
    }
    migraphx::program_parameters params;
    const auto param_names = program->param_shapes.names();
    for (size_t i = 0; i < num_inputs; ++i) {
      // MIGraphX drops graph inputs nothing consumes; skip those.
      if (std::none_of(param_names.begin(), param_names.end(),
                       [&](const char *p) {
                         return std::strcmp(p, input_names[i]) == 0;
                       }))
        continue;
      params.add(input_names[i],
                 migraphx::argument(program->param_shapes[input_names[i]],
                                    input_data[i]));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
      params.add(program->output_params[i].c_str(),
                 migraphx::argument(program->output_shapes[i], output_data[i]));
    }
    program->program.eval(params);
    // eval() only enqueues on MIGraphX's own queue; the EP is not stream-aware,
    // so ORT reads the outputs as soon as Compute returns.
    program->program.experimental_get_context().finish();
    return 0;
  } catch (...) {
    set_error(err, err_size, describe_current_exception("MIGraphX run"));
    return 1;
  }
}

HIPEP_MGX_EXPORT void hipep_mgx_release(hipep_mgx_program *program) {
  delete program;
}

} // extern "C"
