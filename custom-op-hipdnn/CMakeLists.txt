##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##


cmake_minimum_required(VERSION 3.17)
project(
  morphizen-custom-op-hipdnn
  VERSION 1.0.0
  LANGUAGES C CXX)

# TheRock installation root - contains HIP, hipDNN, and other ROCm components
set(THEROCK_DIST "$ENV{THEROCK_DIST}" CACHE PATH "Path to TheRock dist/rocm directory")
if(NOT THEROCK_DIST)
  message(FATAL_ERROR "THEROCK_DIST must be specified. Set it via -DTHEROCK_DIST=<path> or THEROCK_DIST environment variable.")
endif()

# Add TheRock to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")

# Third-party dependencies from TheRock (nlohmann-json is required by hipdnn_sdk)
list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/../../third-party/nlohmann-json/dist/share/cmake")

# Find MIOpen from TheRock
find_package(miopen REQUIRED CONFIG)
find_package(hip REQUIRED CONFIG)
find_package(nlohmann_json CONFIG REQUIRED)

# Find hipDNN from TheRock (optional, for gradual migration)
# find_package(hipdnn_frontend CONFIG)
# find_package(hipdnn_backend CONFIG)
# find_package(hipdnn_data_sdk CONFIG)

find_package (glog CONFIG REQUIRED)
# TODO: do we really need to depends on onnxruntime? it is only needed
# at compile time.
find_package (onnxruntime CONFIG REQUIRED)
set(LIB_NAME morphizen-custom-op-hipdnn)
# Use MIOpen-based implementation (migrated directly in custom_op.cpp)
add_library(${LIB_NAME} STATIC 
  src/main.cpp 
  src/custom_op.hpp 
  src/custom_op.cpp
)
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "hipdnn")
target_include_directories(${LIB_NAME} PRIVATE 
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/hipDNNEP/include
  ${THEROCK_DIST}/include
)
# TODO: remove VAIP_CUSTOM_OP=1
target_compile_definitions(${LIB_NAME} PRIVATE 
  "VAIP_CUSTOM_OP=1"
  "__HIP_PLATFORM_AMD__"
)
target_compile_options(${LIB_NAME} PRIVATE /utf-8)
target_link_libraries(${LIB_NAME} PRIVATE
  hipdnn_proto
  glog::glog
  morphizen::morphizen-core-static
  MIOpen
  hip::host
  nlohmann_json::nlohmann_json
  # Optional: Keep for gradual migration
  # $<$<BOOL:${hipdnn_frontend_FOUND}>:hipdnn_frontend>
  # $<$<BOOL:${hipdnn_backend_FOUND}>:hipdnn_backend>
  # $<$<BOOL:${hipdnn_data_sdk_FOUND}>:hipdnn_data_sdk>
  )
set_target_properties(${LIB_NAME} PROPERTIES
  OUTPUT_NAME "${LIB_NAME}")
target_link_libraries(${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
