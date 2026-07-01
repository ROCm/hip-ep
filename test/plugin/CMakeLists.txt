##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#=============================================================================
# Static compiler-plugin tests
#=============================================================================
# Builds the sample static plugin (available to the registrar) and a unit test
# that runs the static registrar and validates the recorded registry state.
#
# The sample is LINKED into the host only when `sample` is in
# HIPDNN_EP_COMPILER_PLUGINS (see cmake/HipEpPlugins.cmake); the unit test is
# told which case it is in via HIP_EP_EXPECT_SAMPLE and asserts either the
# sample's contributions (selected) or a clean no-op (not selected). This means
# the test passes in the default build (no plugins) and in a plugin-enabled
# build alike.
#
# See docs/design/plugin-interface.md for the broader context.

message(STATUS "Configuring static compiler-plugin tests")

# Sample static plugin — always built (available to the registrar); linked only
# when selected.
add_subdirectory(sample_plugin)

# Plain main()-driven test (no GTest dependency), so it has no dependency the
# public configure does not already provide.
add_executable(test_static_plugins test_static_plugins.cpp)

target_compile_features(test_static_plugins PRIVATE cxx_std_17)

target_link_libraries(test_static_plugins PRIVATE
  LibHipCompiler
)

# Tell the test whether the sample plugin is statically linked into this build
# (i.e. selected via HIPDNN_EP_COMPILER_PLUGINS). dispatchPluginRegistrationsOnce
# comes from LibHipCompiler, which links the sample only when selected.
if("sample" IN_LIST HIPDNN_EP_COMPILER_PLUGINS)
  target_compile_definitions(test_static_plugins PRIVATE HIP_EP_EXPECT_SAMPLE=1)
else()
  target_compile_definitions(test_static_plugins PRIVATE HIP_EP_EXPECT_SAMPLE=0)
endif()

set_target_properties(test_static_plugins PROPERTIES FOLDER "mlir/Test/Plugin")

# Register with CTest.
add_test(NAME StaticPlugins
  COMMAND test_static_plugins
  WORKING_DIRECTORY $<TARGET_FILE_DIR:test_static_plugins>
)
