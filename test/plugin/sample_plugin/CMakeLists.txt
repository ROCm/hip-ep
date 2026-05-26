##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Sample plugin DLL exercising the public plugin ABI surface in CI.
# See docs/design/plugin-extension-api.md for the broader context.

add_library(hip_ep_sample_plugin SHARED
  sample_plugin.cpp
)

# Plugin links against the MLIR libraries it actually uses
# (PassRegistration<>, PassWrapper<>, func::FuncOp). It does NOT link
# LibHipCompiler -- a real vendor plugin is a downstream binary
# consumer of the public hip-compiler artefact, not an internal
# component, so we match that constraint here in the in-tree sample.
#
# This list mirrors the upstream `mlir/examples/standalone/lib/...`
# pattern. Bringing in MLIRPass also brings the MLIRSupport / MLIRIR
# transitive dependencies the pass needs.
target_link_libraries(hip_ep_sample_plugin PRIVATE
  MLIRPass
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
)

# The plugin uses public hip-compiler headers (PluginAPI.h,
# PluginRegistry.h). Both already on the global include path via
# the top-level include_directories(); no per-target setup needed.

# Pull in the LLVM_ATTRIBUTE_WEAK / Compiler.h shim from LLVM.
target_include_directories(hip_ep_sample_plugin PRIVATE
  ${LLVM_INCLUDE_DIRS}
)

# Windows: export hipEpGetPluginInfo without a .def file.
# Mirrors how llvm/examples/Bye/Bye.cpp's CMakeLists handles upstream
# pass plugins on Windows.
set_target_properties(hip_ep_sample_plugin PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "mlir/Test/Plugin"
  # Keep the file name predictable for the test runner; cmake's
  # default would prepend "lib" on non-Windows.
  PREFIX ""
  OUTPUT_NAME "hip_ep_sample_plugin"
)

# C++17 to match the rest of the tree.
target_compile_features(hip_ep_sample_plugin PRIVATE cxx_std_17)
