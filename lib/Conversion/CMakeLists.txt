##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Conversion Libraries")

add_subdirectory(OnnxToHip)
add_subdirectory(HipToLLVM)

# Umbrella conversion library
add_library(HipConversion STATIC
  Passes.cpp
)

target_link_libraries(HipConversion PUBLIC
  OnnxToHip
  HipToLLVM
  MLIRIR
  MLIRPass
)

target_include_directories(HipConversion PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(HipConversion PROPERTIES FOLDER "mlir/Conversion")
