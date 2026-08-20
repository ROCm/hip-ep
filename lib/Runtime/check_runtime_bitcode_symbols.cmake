##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

if(NOT DEFINED LLVM_NM_EXECUTABLE OR NOT DEFINED RUNTIME_BITCODE)
  message(FATAL_ERROR
    "LLVM_NM_EXECUTABLE and RUNTIME_BITCODE are required")
endif()

execute_process(
  COMMAND "${LLVM_NM_EXECUTABLE}" --undefined-only "${RUNTIME_BITCODE}"
  RESULT_VARIABLE llvm_nm_result
  OUTPUT_VARIABLE undefined_symbols
  ERROR_VARIABLE llvm_nm_error
)

if(NOT llvm_nm_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to inspect runtime bitcode with llvm-nm: ${llvm_nm_error}")
endif()

# The default model artifact JIT-links runtime.bc inside the EP process. The
# Windows host does not export the MSVC std::nothrow object or its allocation
# overload, so either reference makes every model fail during global_ctors.
string(REGEX MATCHALL "[^\n]*nothrow[^\n]*" nothrow_symbols
  "${undefined_symbols}")
if(nothrow_symbols)
  message(FATAL_ERROR
    "runtime.bc references unsupported std::nothrow symbols:\n"
    "${nothrow_symbols}")
endif()
