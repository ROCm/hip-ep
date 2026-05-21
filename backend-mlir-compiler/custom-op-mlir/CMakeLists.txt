##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.17)

message(STATUS "Configuring MLIR Custom Op")

# Find dependencies
find_package(glog CONFIG REQUIRED)

# HIP runtime headers + amdhip64 import library are needed by
# MlirCustomOp.cpp's HIPDNN_EP_PERF instrumentation (hipEvent*, hipStream_t,
# hipDeviceSynchronize). THEROCK_DIST is on CMAKE_PREFIX_PATH, so hip::host
# resolves without reordering the parent add_subdirectory sequence.
# Mock builds have no HIP SDK; the PERF path is compiled out via
# BUILD_MOCK_RUNTIME.
if(NOT BUILD_MOCK_RUNTIME)
  find_package(hip CONFIG REQUIRED)
endif()

# Create custom op library
set(LIB_NAME morphizen-custom-op-mlir)
add_library(${LIB_NAME} STATIC
  src/plugin_registration.cpp
  src/MlirCustomOp.cpp
  src/InferenceState.cpp
  src/DimSpecResolver.cpp
)

# Define MORPHIZEN_CUSTOM_OP for conditional compilation
target_compile_definitions(${LIB_NAME} PRIVATE "MORPHIZEN_CUSTOM_OP=1")

if(BUILD_MOCK_RUNTIME)
  target_compile_definitions(${LIB_NAME} PRIVATE "BUILD_MOCK_RUNTIME=1")
endif()

# Link dependencies
target_link_libraries(${LIB_NAME} PRIVATE
  morphizen::morphizen-core-static
  glog::glog
  mlir_compilation_proto  # Shared protobuf metadata library
  # Generated FlatBuffers headers + embedded schema string. Used by the
  # InferenceState to parse the DLL-emitted `inference_get_metadata_json()`
  # output (FB-JSON) into a HipModelMetaInfoT for dynamic-output-shape
  # resolution.
  HipSchemasLib
)
if(NOT BUILD_MOCK_RUNTIME)
  target_link_libraries(${LIB_NAME} PRIVATE hip::host)
endif()

# Include directories
target_include_directories(${LIB_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}/..  # For generated protobuf headers (metadata.pb.h)
  ${CMAKE_SOURCE_DIR}/include     # Shared project headers (timing.h, etc.)
)

# Link to EP target with whole archive to ensure registration
# Only link if target is defined (set by parent morphizen build system)
if(DEFINED morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET AND
   TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(
    ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
    PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>"
  )
else()
  message(WARNING "morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET not defined or not a target - skipping custom-op-mlir registration")
endif()

set_target_properties(${LIB_NAME} PROPERTIES
  FOLDER "mlir"
)
