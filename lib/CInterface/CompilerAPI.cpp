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

// Pack `driver`'s post-compile refined output shapes into `outputs->shapes_buf`
// using the legacy packed-int64 layout. Always writes `*shapes_needed` even
// when the buffer is too small (caller can resize and re-invoke compile, or
// just log a warning — InferOnnxShapes output sizes are bounded by the model
// graph signature, so the buffer size is easy to pick up front).
static void packShapes(const CompilerDriver &driver,
                       CompilationOutputs *outputs) {
  if (!outputs)
    return;
  const auto &shapes = driver.refinedOutputShapes();
  int64_t needed = 1; // num_outputs
  for (const auto &dims : shapes)
    needed += 1 + static_cast<int64_t>(dims.size());
  outputs->shapes_needed = needed;
  if (!outputs->shapes_buf || outputs->shapes_capacity <= 0)
    return;
  int64_t cap = outputs->shapes_capacity;
  int64_t cursor = 0;
  auto put = [&](int64_t v) {
    if (cursor < cap)
      outputs->shapes_buf[cursor] = v;
    ++cursor;
  };
  put(static_cast<int64_t>(shapes.size()));
  for (const auto &dims : shapes) {
    put(static_cast<int64_t>(dims.size()));
    for (int64_t d : dims)
      put(d);
  }
}

// Pack `driver`'s post-compile per-output, per-dim SSA origins into
// `outputs->origins_buf` using the legacy packed-int64 layout (3 slots per
// dim: arg_idx, dim_idx, mult_bits). mult_bits is the IEEE 754 binary64 bit
// pattern of `DimOriginTriple::mult` so the entire payload is one int64 stream.
static void packOrigins(const CompilerDriver &driver,
                        CompilationOutputs *outputs) {
  if (!outputs)
    return;
  const auto &origins = driver.refinedOutputDimOrigins();
  int64_t needed = 1; // num_outputs
  for (const auto &dims : origins)
    needed += 1 + 3 * static_cast<int64_t>(dims.size());
  outputs->origins_needed = needed;
  if (!outputs->origins_buf || outputs->origins_capacity <= 0)
    return;
  int64_t cap = outputs->origins_capacity;
  int64_t cursor = 0;
  auto put = [&](int64_t v) {
    if (cursor < cap)
      outputs->origins_buf[cursor] = v;
    ++cursor;
  };
  put(static_cast<int64_t>(origins.size()));
  for (const auto &dims : origins) {
    put(static_cast<int64_t>(dims.size()));
    for (const auto &info : dims) {
      put(info.arg_idx);
      put(info.dim_idx);
      int64_t mult_bits;
      std::memcpy(&mult_bits, &info.mult, sizeof(double));
      put(mult_bits);
    }
  }
}

extern "C" {

COMPILER_API CompilerErrorCode hip_compile_with_fs(
    const void *input_mlir, size_t input_size, const char *output_path,
    const char *options_json, CompilerError *error, void *fs,
    CompilationOutputs *outputs) {
  // Default-initialise the OUT fields so callers always see a defined
  // value even on error paths (no thread-local stash to consult).
  if (outputs) {
    outputs->shapes_needed = 0;
    outputs->origins_needed = 0;
  }
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

    // After a successful compile, harvest the InferOnnxShapes results
    // (shapes + origins) the driver captured from module attributes.
    // packShapes / packOrigins always write *_needed; only fill the
    // caller's buffer when they passed a non-NULL pointer + positive
    // capacity.
    packShapes(driver, outputs);
    packOrigins(driver, outputs);

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

} // extern "C"
