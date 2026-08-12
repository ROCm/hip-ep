##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipTestPasses STATIC
  TestHipPasses.cpp
)

target_include_directories(HipTestPasses PUBLIC
  ${CMAKE_SOURCE_DIR}/test/lib
)

target_link_libraries(HipTestPasses PUBLIC
  HipDialectIR
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
)
