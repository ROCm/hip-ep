/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../Compiler/CompilerPipeline.h"
#include "compilation_options.pb.h"
#include "morphizen-mlir-compiler/compiler_api.h"
#include "morphizen-mlir-compiler/compiler_types.h"

#include "llvm/ADT/StringRef.h"
#include <google/protobuf/util/json_util.h>

#include <cstring>
#include <string>
#include <vector>

using namespace morphizen::mlir_compiler;

// Version string
static const char* COMPILER_VERSION = "1.0.0";

// Helper: Parse JSON to proto options
static bool parseOptions(const char* options_json,
                         morphizen::mlir_compiler::CompilationOptions& options,
                         std::string& error_message) {
  if (!options_json || strlen(options_json) == 0) {
    // Set defaults
    options.set_opt_level(2);
    options.set_output_mode(morphizen::mlir_compiler::OUTPUT_MODE_DLL);
    return true;
  }

  // Parse JSON to proto
  google::protobuf::util::JsonParseOptions parse_opts;
  parse_opts.ignore_unknown_fields = false;

  auto status = google::protobuf::util::JsonStringToMessage(
      options_json, &options, parse_opts);

  if (!status.ok()) {
    error_message =
        "Failed to parse options JSON: " + std::string(status.message());
    return false;
  }

  // Validate opt_level range
  if (options.opt_level() < 0 || options.opt_level() > 3) {
    options.set_opt_level(2); // Clamp to valid range
  }

  // Validate output_mode (set default if unspecified)
  if (options.output_mode() ==
      morphizen::mlir_compiler::OUTPUT_MODE_UNSPECIFIED) {
    options.set_output_mode(morphizen::mlir_compiler::OUTPUT_MODE_DLL);
  }

  return true;
}

// Helper: Set error message
static void setError(CompilerError* error, const std::string& message) {
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

COMPILER_API CompilerErrorCode morphizen_mlir_compile(const void* input_mlir,
                                                      size_t input_size,
                                                      const char* output_path,
                                                      const char* options_json,
                                                      CompilerError* error) {
  // Validate inputs
  if (!input_mlir || input_size == 0 || !output_path) {
    setError(
        error,
        "Invalid input: input_mlir, input_size, and output_path must be valid");
    return COMPILER_ERROR_INVALID_INPUT;
  }

  try {
    // Parse JSON to proto options
    morphizen::mlir_compiler::CompilationOptions options;
    std::string parse_error;
    if (!parseOptions(options_json, options, parse_error)) {
      setError(error, parse_error);
      return COMPILER_ERROR_INVALID_INPUT;
    }

    // Create compiler pipeline
    CompilerPipeline pipeline;

    // Compile (pass proto directly)
    // Binary-safe: uses input_size, not null terminator
    // Zero-copy: StringRef references the input data directly
    std::string error_message;
    llvm::StringRef input_ref(static_cast<const char*>(input_mlir), input_size);
    std::string output_str(output_path);

    bool success =
        pipeline.compile(input_ref, output_str, options, error_message);

    if (!success) {
      setError(error, error_message);
      return COMPILER_ERROR_COMPILATION_FAILED;
    }

    return COMPILER_SUCCESS;

  } catch (const std::exception& ex) {
    setError(error, std::string("Exception: ") + ex.what());
    return COMPILER_ERROR_INTERNAL;
  } catch (...) {
    setError(error, "Unknown exception occurred");
    return COMPILER_ERROR_INTERNAL;
  }
}

COMPILER_API const char* morphizen_mlir_get_version(void) {
  return COMPILER_VERSION;
}

} // extern "C"
