##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#=============================================================================
# Plugin loader tests
#=============================================================================
# Builds a small sample plugin DLL exposing the public ABI from
# include/hip/Compiler/PluginAPI.h, and a unit test that loads it and
# validates the loader's behaviour.
#
# See docs/design/plugin-interface.md for the broader context.

message(STATUS "Configuring plugin loader tests")

# Sample plugin DLL — built unconditionally so CTest can always run
# the loader test.
add_subdirectory(sample_plugin)

# Plain main()-driven test (no GTest dependency), so it has no dependency the
# public configure does not already provide.
add_executable(test_plugin_loader test_plugin_loader.cpp)

target_compile_features(test_plugin_loader PRIVATE cxx_std_17)

target_link_libraries(test_plugin_loader PRIVATE
  LibHipCompiler
)

# Pass the absolute path of the sibling sample plugin into the test
# at compile time so it does not depend on cwd / PATH at run time.
target_compile_definitions(test_plugin_loader PRIVATE
  "HIP_EP_SAMPLE_PLUGIN_PATH=\"$<TARGET_FILE:hip_ep_sample_plugin>\""
)

# Make sure the sample plugin is built before the test runs.
add_dependencies(test_plugin_loader hip_ep_sample_plugin)

set_target_properties(test_plugin_loader PROPERTIES FOLDER "mlir/Test/Plugin")

# Register with CTest. WORKING_DIRECTORY is the build directory of the
# test exe so the sample plugin DLL sits next to it on Windows (where
# the loader resolves dependent DLLs from the EXE's directory by default).
add_test(NAME PluginLoader
  COMMAND test_plugin_loader
  WORKING_DIRECTORY $<TARGET_FILE_DIR:test_plugin_loader>
)
