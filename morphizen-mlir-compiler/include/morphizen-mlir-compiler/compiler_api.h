/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef MORPHIZEN_MLIR_COMPILER_API_H
#define MORPHIZEN_MLIR_COMPILER_API_H

#include "compiler_types.h"

/* Export macro for DLL visibility */
#ifdef _WIN32
#  ifdef MORPHIZEN_MLIR_COMPILER_EXPORTS
#    define COMPILER_API __declspec(dllexport)
#  else
#    define COMPILER_API __declspec(dllimport)
#  endif
#else
#  define COMPILER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Main compilation entry point.
 *
 * Compiles MLIR input (text or bytecode) to DLL/object/IR file.
 *
 * @param input_mlir   Input MLIR data (text or bytecode, binary-safe)
 * @param input_size   Size of input data in bytes
 * @param output_path  Output file path (DLL/object/IR depending on options)
 * @param options_json Compilation options as JSON string (proto-serialized).
 *                     Example: {"opt_level": 2, "output_mode":
 * "OUTPUT_MODE_DLL"} Pass NULL for default options (opt_level=2,
 * output_mode=DLL).
 * @param error        Error information output (can be NULL)
 * @return             COMPILER_SUCCESS or error code
 */
COMPILER_API CompilerErrorCode morphizen_mlir_compile(const void* input_mlir,
                                                      size_t input_size,
                                                      const char* output_path,
                                                      const char* options_json,
                                                      CompilerError* error);

/**
 * Get compiler version string.
 *
 * @return Static version string (e.g., "1.0.0")
 */
COMPILER_API const char* morphizen_mlir_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* MORPHIZEN_MLIR_COMPILER_API_H */
