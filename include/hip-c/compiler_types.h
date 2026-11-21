/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_TYPES_H
#define HIP_COMPILER_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes for compiler operations */
typedef enum {
  COMPILER_SUCCESS = 0,
  COMPILER_ERROR_INVALID_INPUT = 1,
  COMPILER_ERROR_PARSE_FAILED = 2,
  COMPILER_ERROR_PASS_FAILED = 3,
  COMPILER_ERROR_TRANSLATION_FAILED = 4,
  COMPILER_ERROR_LINKING_FAILED = 5,
  COMPILER_ERROR_IO_ERROR = 6,
  COMPILER_ERROR_COMPILATION_FAILED = 7,
  COMPILER_ERROR_VALIDATION_FAILED = 8,
  COMPILER_ERROR_INTERNAL = 99
} CompilerErrorCode;

/* Error information (DLL-safe, fixed-size buffer) */
#define COMPILER_ERROR_MESSAGE_SIZE 1024
typedef struct {
  CompilerErrorCode code;
  char message[COMPILER_ERROR_MESSAGE_SIZE];
} CompilerError;

#ifdef __cplusplus
}
#endif

#endif /* HIP_COMPILER_TYPES_H */
