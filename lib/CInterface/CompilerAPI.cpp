/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/CompilerDriver.h"
#include "hip/compiler_api.h"
#include "hip/compiler_types.h"
#include "hip/flatbuffers_json.h"

#include "morphizen-foundation/file_io.hpp"

#include "compilation_options_schema.h"

#include "llvm/ADT/StringRef.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

// Forward decls from lib/Conversion/OnnxToHip/InferOnnxShapes.cpp. We
// avoid including the full MLIR header from this C-ABI translation
// unit; both translation units link into hip-compiler.dll so the
// symbols resolve at link time. Both accessors return data captured by
// `inferOnnxShapes` WHILE the function is still `func::FuncOp` (before
// HipToLLVM converts it to `llvm.func`); reading them post-compile gets
// the most recent compile's snapshot on the same thread.
//
// Keep `DimOriginInfo` byte-compatible with `lib/Conversion/OnnxToHip/
// OnnxToHipUtils.h` — two int64s + a double in declared order. Any change
// there must update here.
namespace mlir {
namespace hip {
struct DimOriginInfo {
  int64_t arg_idx;
  int64_t dim_idx;
  double mult;
};
const std::vector<std::vector<int64_t>> &getInferredOutputShapes();
const std::vector<std::vector<DimOriginInfo>> &getInferredOutputOrigins();
} // namespace hip
} // namespace mlir

#ifdef HIPDNN_GRAPH_RUNTIME_AVAILABLE
#include "../HipDNNGraphRuntime/hipdnn_graph_runtime.h"
#endif

using namespace hip::compiler;

static const char *COMPILER_VERSION = "1.0.0";

// Parse JSON into CompilationOptionsT (defined in
// schemas/compilation_options.fbs). Key fields:
//   opt_level          — LLVM optimization level 0-3 (default 2)
//   output_mode        — DLL or LLVM_IR (default DLL)
//   constants_file     — externalized weights filename (default
//   "constants.bin") skip_constant_data — skip writing constant bytes (default
//   false)
static bool parseOptions(const char *options_json,
                         mlir::hip::CompilationOptionsT &opts,
                         std::string &error_message) {
  if (!options_json || strlen(options_json) == 0)
    return true;

  return mlir::hip::fromJson<mlir::hip::CompilationOptionsT>(
      options_json, mlir::hip::k_compilation_options_schema(), opts,
      error_message);
}

static void setError(CompilerError *error, const std::string &message) {
  if (error) {
    size_t len = message.length();
    if (len >= sizeof(error->message)) {
      len = sizeof(error->message) - 1;
    }
    std::memcpy(error->message, message.c_str(), len);
    error->message[len] = '\0';
  }
}

extern "C" {

COMPILER_API CompilerErrorCode hip_compile_with_fs(
    const void *input_mlir, size_t input_size, const char *output_path,
    const char *options_json, CompilerError *error, void *fs) {
  if (!input_mlir || input_size == 0 || !output_path) {
    setError(error, "Invalid input: input_mlir, input_size, and output_path "
                    "must be valid");
    return COMPILER_ERROR_INVALID_INPUT;
  }
  if (!fs) {
    setError(error, "Invalid input: fs (FileSystem*) must be non-null");
    return COMPILER_ERROR_INVALID_INPUT;
  }

  try {
    mlir::hip::CompilationOptionsT options;
    std::string parse_error;
    if (!parseOptions(options_json, options, parse_error)) {
      setError(error, parse_error);
      return COMPILER_ERROR_INVALID_INPUT;
    }

    CompilerDriver driver;
    driver.setFileSystem(static_cast<morphizen::FileSystem *>(fs));

#ifdef HIPDNN_GRAPH_RUNTIME_AVAILABLE
    void *hipdnn_handle = hipdnn_graph_create_handle();
    if (hipdnn_handle)
      driver.setHipdnnHandle(hipdnn_handle);
#endif

    std::string error_message;
    llvm::StringRef input_ref(static_cast<const char *>(input_mlir),
                              input_size);
    std::string output_str(output_path);

    bool success =
        driver.compile(input_ref, output_str, options, error_message);

    if (!success) {
      setError(error, error_message);
      return COMPILER_ERROR_COMPILATION_FAILED;
    }

#ifdef HIPDNN_GRAPH_RUNTIME_AVAILABLE
    if (hipdnn_handle) {
      auto graphs = driver.getCompiledGraphs();
      if (graphs && !graphs->empty()) {
        void *registry = hipdnn_graph_registry_create();
        for (auto &[name, graph] : *graphs) {
          int graph_id = std::stoi(name.str().substr(strlen("hipdnn_graph_")));
          hipdnn_graph_registry_store(registry, graph_id, graph.release());
        }
        hipdnn_graph_set_default_registry(registry);
        hipdnn_graph_set_default_handle(hipdnn_handle);
      }
    }
#endif

    return COMPILER_SUCCESS;

  } catch (const std::exception &ex) {
    setError(error, std::string("Exception: ") + ex.what());
    return COMPILER_ERROR_INTERNAL;
  } catch (...) {
    setError(error, "Unknown exception occurred");
    return COMPILER_ERROR_INTERNAL;
  }
}

COMPILER_API const char *hip_get_version(void) { return COMPILER_VERSION; }

COMPILER_API int64_t hip_get_last_compile_output_shapes(int64_t *out_buffer,
                                                        int64_t buffer_size) {
  // Layout: [num_outputs, num_dims_0, d0_0, d0_1, ..., num_dims_1, ...].
  // Two-call discovery: pass NULL to query size, then allocate, then call
  // again with the buffer.
  const auto &shapes = mlir::hip::getInferredOutputShapes();
  int64_t needed = 1; // for num_outputs
  for (const auto &dims : shapes)
    needed += 1 + static_cast<int64_t>(dims.size()); // for num_dims + dims
  if (!out_buffer)
    return needed;
  if (buffer_size < needed)
    return needed; // truncation indicator
  int64_t cursor = 0;
  out_buffer[cursor++] = static_cast<int64_t>(shapes.size());
  for (const auto &dims : shapes) {
    out_buffer[cursor++] = static_cast<int64_t>(dims.size());
    for (int64_t d : dims)
      out_buffer[cursor++] = d;
  }
  return needed;
}

COMPILER_API int64_t hip_get_last_compile_output_dim_origins(
    int64_t *out_buffer, int64_t buffer_size) {
  // Layout (per output, in declaration order):
  //   [num_outputs,
  //    num_dims_0, arg_0_0, dim_0_0, mult_bits_0_0,
  //                arg_0_1, dim_0_1, mult_bits_0_1, ...,
  //    num_dims_1, arg_1_0, dim_1_0, mult_bits_1_0, ...,
  //    ...]
  // Each (arg, dim, mult_bits) triple is the function-arg-position +
  // arg-dim-index + IEEE 754 binary64 bit pattern of the scalar multiplier
  // for the SSA trace of the corresponding output dim. Runtime computes
  // the output dim's runtime value as
  // `round(inputs[arg].shape[dim] * bit_cast<double>(mult_bits))`.
  // `mult == 1.0` is the identity passthrough (most LLM dynshape outputs);
  // `mult == 1/K` covers Reshape-induced spatial mergers (e.g. Qwen
  // vision's `num_patches -> num_patches/4` patch merger contributes
  // mult=0.25); `mult > 1.0` is reserved for future multiply-by-K
  // (spatial upsamplers; trace rule not yet enabled).
  //
  // `(-1, -1, 1.0)` means no traceable origin — the EP either resolves
  // the dim via a different DimSource priority or fails the compile
  // loudly.
  //
  // Bit-cast (memcpy) is chosen over a separate float buffer to keep the
  // C ABI a single int64 stream; sizeof(double) == sizeof(int64_t).
  // Same two-call discovery pattern as `hip_get_last_compile_output_shapes`.
  const auto &origins = mlir::hip::getInferredOutputOrigins();
  int64_t needed = 1; // num_outputs
  for (const auto &dims : origins)
    needed += 1 + 3 * static_cast<int64_t>(dims.size());
  if (!out_buffer)
    return needed;
  if (buffer_size < needed)
    return needed;
  int64_t cursor = 0;
  out_buffer[cursor++] = static_cast<int64_t>(origins.size());
  for (const auto &dims : origins) {
    out_buffer[cursor++] = static_cast<int64_t>(dims.size());
    for (const auto &info : dims) {
      out_buffer[cursor++] = info.arg_idx;
      out_buffer[cursor++] = info.dim_idx;
      int64_t mult_bits;
      std::memcpy(&mult_bits, &info.mult, sizeof(double));
      out_buffer[cursor++] = mult_bits;
    }
  }
  return needed;
}

} // extern "C"
