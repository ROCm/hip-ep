##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(${CMAKE_CURRENT_SOURCE_DIR}/../morphizen-core/cmake/find_onnxruntime.cmake)
set(LIB_NAME morphizen-ort-api-ext)
add_library(${LIB_NAME} STATIC
  src/morphizen-ort-api-ext.cpp
  src/ort-api-version.cpp
)
target_sources(${LIB_NAME} PUBLIC
  FILE_SET modules_public TYPE HEADERS FILES
  ${CMAKE_CURRENT_LIST_DIR}/include/morphizen/morphizen-ort-api-ext.hpp
  ${CMAKE_CURRENT_LIST_DIR}/include/morphizen/ort-api-version.hpp
)
target_include_directories(${LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${MORPHIZEN_ORT_API_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(${LIB_NAME} PUBLIC cxx_std_17)
target_compile_options(${LIB_NAME} PRIVATE ${MORPHIZEN_COMPILER_OPTIONS})
target_link_libraries(${LIB_NAME}
  PUBLIC
  onnxruntime::onnxruntime
  Microsoft.GSL::GSL
  glog::glog
  PRIVATE
  morphizen-utils
)
