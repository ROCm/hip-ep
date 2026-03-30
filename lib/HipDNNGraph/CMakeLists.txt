##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building HipDNNGraph library")

add_library(HipDNNGraph STATIC
  HipDNNGraph.cpp
)

# PUBLIC: consumers need HipDNNGraph.h (forward-declares only, no TheRock types).
target_include_directories(HipDNNGraph PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(HipDNNGraph PUBLIC
  MLIRIR
  MLIRSupport
  LLVMSupport
)

set_target_properties(HipDNNGraph PROPERTIES FOLDER "mlir/HipDNNGraph")
