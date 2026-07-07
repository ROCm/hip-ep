##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##


cmake_minimum_required(VERSION 3.17)
project(
  morphizen-pass_init
  VERSION 1.0.0
  LANGUAGES C CXX)

add_library(morphizen-pass-init STATIC src/morphizen_pass_init.cpp)
set_target_properties (morphizen-pass-init
  PROPERTIES
  OUTPUT_NAME morphizen-pass-init)
add_library (morphizen::pass-init ALIAS morphizen-pass-init)
target_link_libraries(morphizen-pass-init PRIVATE morphizen-core-static)
# TODO: morphizen pass naming convention: morphizen-<pass_a_b_c>
# overwrite the default OUTPUT_NAME in morphizen_add_library
target_compile_definitions(morphizen-pass-init
  PRIVATE -DOUTPUT_NAME="morphizen-pass-init")

if(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,morphizen-pass-init>)
endif(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})

set_target_properties(morphizen-pass-init
  PROPERTIES
  FOLDER "morphizen/passes")
