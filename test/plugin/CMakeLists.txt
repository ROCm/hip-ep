##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#=============================================================================
# Static compiler-plugin tests: the sample plugin (available to the registrar)
# + a unit test that runs the registrar and checks the recorded registry state.
# The sample is linked only when `sample` is in HIPDNN_EP_COMPILER_PLUGINS;
# HIP_EP_EXPECT_SAMPLE tells the test which case to assert, so it passes in the
# default (no-plugin) build and a plugin-enabled build alike.
#=============================================================================

message(STATUS "Configuring static compiler-plugin tests")

# Always built (available to the registrar); linked only when selected.
add_subdirectory(sample_plugin)

# Plain main()-driven test (no GTest dependency).
add_executable(test_static_plugins test_static_plugins.cpp)

target_compile_features(test_static_plugins PRIVATE cxx_std_17)

target_link_libraries(test_static_plugins PRIVATE
  LibHipCompiler
)

# Tell the test whether the sample plugin is linked into this build.
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
