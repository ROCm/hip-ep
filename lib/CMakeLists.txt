##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_subdirectory(Support)
add_subdirectory(Dialect)
option(BUILD_HIPDNN_GRAPH "Build hipDNN fused-graph support" OFF)
set(HIPDNN_AVAILABLE OFF)
if(NOT BUILD_MOCK_RUNTIME AND BUILD_HIPDNN_GRAPH)
  set(HIPDNN_AVAILABLE ON)
  add_subdirectory(HipDNNGraph)
  add_subdirectory(HipDNNGraphRuntime)
endif()
add_subdirectory(Conversion)
add_subdirectory(Runtime)
if(NOT BUILD_MOCK_RUNTIME)
  add_subdirectory(MemoryManager)
endif()
add_subdirectory(Target/LLVM)
add_subdirectory(Compiler)
add_subdirectory(CInterface)
