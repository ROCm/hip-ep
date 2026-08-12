##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_subdirectory(hip-mlir-opt)
add_subdirectory(hip-compiler)

# Offline generator for GQA autotune LUT rows. Host-only, no GPU dependency.
add_subdirectory(gqa-autotune-lut)

# hip-inspect parses the IR with `llvm::parseBitcodeFile` and reads
# the `@__metadata_json` global directly -- no JIT, no GPU dependency,
# so it builds whenever BUILD_HIP_TOOLS is on.
add_subdirectory(hip-inspect)

# hip-test is dual-format (bitcode via LlvmIrJit, native .dll/.so via
# morphizen::Plugin), using the same loaders the EP uses at runtime. Those
# live in `morphizen-custom-op-mlir`, registered only when BUILD_EP is on, so
# gate on the target to keep a BUILD_HIP_TOOLS-only configure valid.
if(TARGET morphizen-custom-op-mlir)
  add_subdirectory(hip-test)
else()
  message(STATUS "hip-test: morphizen-custom-op-mlir target absent "
                 "(BUILD_EP=OFF); skipping")
endif()
