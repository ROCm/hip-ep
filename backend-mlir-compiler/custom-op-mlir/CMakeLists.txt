##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.17)

message(STATUS "Configuring MLIR Custom Op")

# Find dependencies
find_package(glog CONFIG REQUIRED)

# Create custom op library
set(LIB_NAME morphizen-custom-op-mlir)
add_library(${LIB_NAME} STATIC
  src/main.cpp
  src/MlirCustomOp.cpp
  src/MetadataParser.cpp
  src/ArtifactLoader.cpp
  src/NativeDllLoader.cpp
  src/LlvmJitLoader.cpp
  src/InferenceState.cpp
  src/TensorMarshaler.cpp
)

# Define MORPHIZEN_CUSTOM_OP for conditional compilation
target_compile_definitions(${LIB_NAME} PRIVATE "MORPHIZEN_CUSTOM_OP=1")

# Link dependencies
target_link_libraries(${LIB_NAME} PRIVATE
  morphizen::morphizen-core-static
  glog::glog
  mlir_compilation_proto  # Protobuf metadata library
)

# Include directories
target_include_directories(${LIB_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}/..  # For generated protobuf headers (mlir_compilation.pb.h)
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
