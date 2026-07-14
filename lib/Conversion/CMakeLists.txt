##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Conversion Libraries")

add_subdirectory(OnnxToHip)
add_subdirectory(OnnxToHipsr)
add_subdirectory(OnnxToHipDNN)
add_subdirectory(HipToLLVM)

# Umbrella conversion library
add_library(HipConversion STATIC
  Passes.cpp
)

target_link_libraries(HipConversion PUBLIC
  OnnxToHip
  OnnxToHipsr
  OnnxToHipDNN
  HipToLLVM
  MLIRIR
  MLIRPass
)

set_target_properties(HipConversion PROPERTIES FOLDER "mlir/Conversion")
