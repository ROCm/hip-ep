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

using namespace hip::compiler;

static const char *COMPILER_VERSION = "1.0.0";

// Parse JSON into CompilationOptionsT (defined in
// schemas/compilation_options.fbs). Key fields:
//   opt_level      — LLVM optimization level 0-3 (default 2)
//   output_mode    — DLL or LLVM_IR (default DLL)
//   constants_file — externalized weights filename (default "constants.bin")
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
    setError(
        error,
        "Invalid input: input_mlir, input_size, and output_path must be valid");
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
