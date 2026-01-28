##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

project(
  morphizen-custom-op-generic
  VERSION 1.0.0
  LANGUAGES C CXX)
set(LIB_NAME "morphizen-custom-op-generic")
add_library(
  ${LIB_NAME} STATIC
  src/custom_op_generic.hpp
  src/custom_op_generic.cpp
  src/custom_op_generic_main.cpp
)
find_package (glog CONFIG REQUIRED)
set(LIB_NAME morphizen-custom-op-generic)
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "morphizen/custom-ops")
target_link_libraries(${LIB_NAME} PRIVATE
  glog::glog
  morphizen::morphizen-core-static
  )

set_target_properties(${LIB_NAME} PROPERTIES
  OUTPUT_NAME "${LIB_NAME}")
target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
