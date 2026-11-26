##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# MIGraphX backend plugin, loaded by the EP at run time from the directory that
# holds the EP library. Only this target links MIGraphX.
find_package(migraphx REQUIRED)

add_library(hipep_migraphx SHARED hipep_migraphx_plugin.cpp)
target_compile_features(hipep_migraphx PRIVATE cxx_std_17)
target_include_directories(hipep_migraphx PRIVATE
  ${CMAKE_SOURCE_DIR}/morphizen/ort-bridge/src)
target_link_libraries(hipep_migraphx PRIVATE migraphx::c)

# The EP opens "<ep dir>/hipep_migraphx.so" (no "lib" prefix) on Linux and
# "<ep dir>/hipep_migraphx.dll" on Windows.
set_target_properties(hipep_migraphx PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")

install(TARGETS hipep_migraphx
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
