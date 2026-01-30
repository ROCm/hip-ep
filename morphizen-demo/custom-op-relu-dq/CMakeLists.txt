##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##


cmake_minimum_required(VERSION 3.17)
project(
  morphizen-custom-op-relu-dq
  VERSION 1.0.0
  LANGUAGES C CXX)

find_package (glog CONFIG REQUIRED)
# TODO: do we really need to depends on onnxruntime? it is only needed
# at compile time.
find_package (onnxruntime CONFIG REQUIRED)
set(LIB_NAME morphizen-custom-op-relu-dq)
add_library(${LIB_NAME} STATIC src/main.cpp src/custom_op.hpp src/custom_op.cpp)
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "relu-dq")
target_include_directories(${LIB_NAME} PRIVATE include)
target_compile_definitions(${LIB_NAME} PRIVATE "MORPHIZEN_CUSTOM_OP=1")
target_link_libraries(${LIB_NAME} PRIVATE
  relu_dq_proto
  glog::glog
  morphizen::morphizen-core-static
  )
set_target_properties(${LIB_NAME} PROPERTIES
  OUTPUT_NAME "${LIB_NAME}")
target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
