##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_subdirectory(hip-mlir-opt)
add_subdirectory(hip-compiler)

# hip-test-bc / hip-inspect-bc are the bitcode-era replacements for the
# retired hip-test-dll / hip-inspect-dll tools. They consume a `.bc`
# artifact emitted by hip-compiler:
#   * hip-inspect-bc parses the IR with `llvm::parseBitcodeFile` and
#     reads the `@__metadata_json` global directly -- no JIT, no GPU
#     dependency, works on a bare dev box, so it builds whenever
#     BUILD_HIP_TOOLS is on.
#   * hip-test-bc goes through the same BitcodeJIT loader the EP uses
#     at runtime; that loader (and its WHOLE_ARCHIVE-linked launcher
#     symbols) live in `morphizen-custom-op-mlir`, which is only
#     registered when BUILD_EP is on. Gate on the target so a
#     BUILD_HIP_TOOLS-only configure doesn't see an undefined dep.
add_subdirectory(hip-inspect-bc)
if(TARGET morphizen-custom-op-mlir)
  add_subdirectory(hip-test-bc)
else()
  message(STATUS "hip-test-bc: morphizen-custom-op-mlir target absent "
                 "(BUILD_EP=OFF); skipping")
endif()
