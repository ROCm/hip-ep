##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building hipdnn_graph_runtime shared library")

add_library(hipdnn_graph_runtime SHARED
  hipdnn_graph_runtime.cpp
)

target_compile_definitions(hipdnn_graph_runtime PRIVATE
  HIPDNN_GRAPH_RUNTIME_EXPORTS
  __HIP_PLATFORM_AMD__
  HIP_INCLUDE_HIP_HIP_VECTOR_TYPES_H
  HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H
  HIP_INCLUDE_HIP_AMD_DETAIL_HIP_VECTOR_TYPES_H
)

target_include_directories(hipdnn_graph_runtime PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_include_directories(hipdnn_graph_runtime BEFORE PRIVATE
  $ENV{THEROCK_DIST}/include
  $ENV{THEROCK_DIST}/include/hipdnn
  $ENV{THEROCK_DIST}/include/hipdnn/frontend
  $ENV{THEROCK_DIST}/include/hipdnn/backend
  $ENV{THEROCK_DIST}/include/hipdnn/data_sdk
  $ENV{THEROCK_DIST}/include/hipdnn/flatbuffers_sdk
  $ENV{THEROCK_DIST}/include/hipdnn/data_sdk/hipdnn_data_sdk/data_objects/v25_9_23
)

if(MSVC)
  target_compile_options(hipdnn_graph_runtime PRIVATE /GR-)
  set_source_files_properties(hipdnn_graph_runtime.cpp PROPERTIES
    COMPILE_OPTIONS "/GR;/permissive")
endif()

# HipDNNGraph for graph execution, Runtime includes for RuntimeState
target_link_libraries(hipdnn_graph_runtime PRIVATE
  HipDNNGraph
)

# hipdnn_backend import library for hipDNN API calls
find_library(HIPDNN_BACKEND_LIB hipdnn_backend
  PATHS $ENV{THEROCK_DIST}/lib
  NO_DEFAULT_PATH
)
if(HIPDNN_BACKEND_LIB)
  target_link_libraries(hipdnn_graph_runtime PRIVATE ${HIPDNN_BACKEND_LIB})
endif()

# HIP runtime for workspace management (hipMalloc etc.)
find_library(AMDHIP64_LIB amdhip64
  PATHS $ENV{THEROCK_DIST}/lib
  NO_DEFAULT_PATH
)
if(AMDHIP64_LIB)
  target_link_libraries(hipdnn_graph_runtime PRIVATE ${AMDHIP64_LIB})
endif()

set_target_properties(hipdnn_graph_runtime PROPERTIES
  FOLDER "mlir/HipDNNGraphRuntime"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

install(TARGETS hipdnn_graph_runtime
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
