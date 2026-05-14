##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_subdirectory(hip-mlir-opt)
add_subdirectory(hip-compiler)

# tools/hip-test-dll and tools/hip-inspect-dll loaded a compiled per-model
# DLL and probed its symbols. After the switch to bitcode (see
# backend-mlir-compiler/custom-op-mlir/src/BitcodeJIT.h) the per-model
# artifact is LLVM bitcode, not a PE, so the DLL-loader tools no longer
# match the artifact format. They are retired here -- a future
# BitcodeJIT-based replacement can live alongside hip-compiler in the
# same tools/ tree.
# add_subdirectory(hip-test-dll)
# add_subdirectory(hip-inspect-dll)
