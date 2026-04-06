##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_subdirectory(Dialect)
if(NOT BUILD_MOCK_RUNTIME)
  add_subdirectory(HipDNNGraph)
  add_subdirectory(HipDNNGraphRuntime)
endif()
add_subdirectory(Conversion)
add_subdirectory(Runtime)
add_subdirectory(Target/LLVM)
add_subdirectory(Compiler)
add_subdirectory(CInterface)
