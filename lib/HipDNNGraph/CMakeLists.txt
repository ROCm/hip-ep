##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building HipDNNGraph library")

add_library(HipDNNGraph STATIC
  HipDNNGraph.cpp
)

target_compile_definitions(HipDNNGraph PRIVATE
  __HIP_PLATFORM_AMD__
  HIP_INCLUDE_HIP_HIP_VECTOR_TYPES_H
  HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H
  HIP_INCLUDE_HIP_AMD_DETAIL_HIP_VECTOR_TYPES_H
)

# PUBLIC: consumers need HipDNNGraph.h (forward-declares only, no TheRock types).
# PRIVATE + BEFORE: TheRock headers must appear before MLIR/LLVM transitive
# includes so that TheRock's FlatBuffers 25.9.23 takes precedence over
# the project's 25.12.19 (hipDNN generated headers require exact match).
target_include_directories(HipDNNGraph PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(HipDNNGraph BEFORE PRIVATE
  $ENV{THEROCK_DIST}/include
  $ENV{THEROCK_DIST}/include/hipdnn
  $ENV{THEROCK_DIST}/include/hipdnn/frontend
  $ENV{THEROCK_DIST}/include/hipdnn/backend
  $ENV{THEROCK_DIST}/include/hipdnn/data_sdk
  $ENV{THEROCK_DIST}/include/hipdnn/flatbuffers_sdk
  $ENV{THEROCK_DIST}/include/hipdnn/data_sdk/hipdnn_data_sdk/data_objects/v25_9_23
)

# hipDNN frontend requires RTTI (dynamic_cast, typeid); MLIR libraries use /GR-.
# /permissive relaxes strict conformance so MSVC accepts HIP __attribute__ syntax
# (overrides /permissive- set by HandleLLVMOptions).
if(MSVC)
  target_compile_options(HipDNNGraph PRIVATE /GR-)
  set_source_files_properties(HipDNNGraph.cpp PROPERTIES
    COMPILE_OPTIONS "/GR;/permissive")
endif()

target_link_libraries(HipDNNGraph PUBLIC
  MLIRIR
  MLIRSupport
  LLVMSupport
)

# hipDNN backend import library -- the frontend is header-only and instantiates
# templates that call into hipdnn_backend.dll.  Since HipDNNGraph is STATIC,
# consumers must transitively link this import lib.
find_library(HIPDNN_BACKEND_LIB_GRAPH hipdnn_backend
  PATHS $ENV{THEROCK_DIST}/lib
  NO_DEFAULT_PATH
)
if(HIPDNN_BACKEND_LIB_GRAPH)
  target_link_libraries(HipDNNGraph PUBLIC ${HIPDNN_BACKEND_LIB_GRAPH})
endif()

# HIP runtime -- needed for hipMalloc/hipFree (workspace allocation in Compile).
find_library(AMDHIP64_LIB_GRAPH amdhip64
  PATHS $ENV{THEROCK_DIST}/lib
  NO_DEFAULT_PATH
)
if(AMDHIP64_LIB_GRAPH)
  target_link_libraries(HipDNNGraph PUBLIC ${AMDHIP64_LIB_GRAPH})
else()
  message(WARNING "amdhip64 not found for HipDNNGraph -- hipMalloc/hipFree will fail to link")
endif()

set_target_properties(HipDNNGraph PROPERTIES FOLDER "mlir/HipDNNGraph")
