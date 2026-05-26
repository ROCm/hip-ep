##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Sample plugin DLL exercising the PR-1 plugin ABI surface in CI.
# See docs/design/plugin-extension-api.md for the broader context.

add_library(hip_ep_sample_plugin SHARED
  sample_plugin.cpp
)

# The plugin only needs the public PluginAPI.h header. We pick it up
# via the global include_directories() from the top-level CMakeLists.
# We do NOT link against LibHipCompiler — the plugin is a tiny ABI
# consumer, not a hip-compiler internal.

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
