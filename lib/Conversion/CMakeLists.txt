##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Conversion Libraries")

add_subdirectory(OnnxToHip)
add_subdirectory(OnnxToHipDNN)
add_subdirectory(HipToLLVM)

## Umbrella aggregator: an INTERFACE target that re-exposes the per-subsystem
## conversion libraries as a single dependency. The TableGen-generated
## `mlir::hip::registerHipConversionPasses()` (defined in
## `include/hip/Conversion/Passes.h.inc`) replaces the previous hand-written
## lambda registrations that lived in `Passes.cpp`.
add_library(HipConversion INTERFACE)

target_link_libraries(HipConversion INTERFACE
  OnnxToHip
  OnnxToHipDNN
  HipToLLVM
)
