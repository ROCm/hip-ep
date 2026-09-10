##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_subdirectory(hip-mlir-opt)
add_subdirectory(hip-compiler)

# hip-rocmlir-compiler hands serialized IR to rocMLIR's high-level pipeline
# inside librockCompiler.so, loaded with RTLD_LOCAL so the .so's own embedded
# copy of LLVM/MLIR stays out of the global symbol namespace. Neither the
# library nor that scoping exists on Windows, so the tool is Linux-only.
if(NOT WIN32)
  add_subdirectory(hip-rocmlir-compiler)
endif()

# hip-inspect parses the IR with `llvm::parseBitcodeFile` and reads
# the `@__metadata_json` global directly -- no JIT, no GPU dependency.
add_subdirectory(hip-inspect)

# hip-test is dual-format (bitcode via LlvmIrJit, native .dll/.so via
# morphizen::Plugin), using the same loaders the EP uses at runtime.
add_subdirectory(hip-test)
