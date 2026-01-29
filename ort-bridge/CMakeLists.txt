##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(${CMAKE_CURRENT_SOURCE_DIR}/../morphizen-core/cmake/find_onnxruntime.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check-ort-version.cmake)

# Compile-time backend selection: C++ code needs to know which backends are available
# to set the correct default backend without runtime overhead
if(morphizen_ENABLE_ONNX_BACKEND)
  add_compile_definitions(MORPHIZEN_ENABLE_ONNX_BACKEND=1)
else()
  add_compile_definitions(MORPHIZEN_ENABLE_ONNX_BACKEND=0)
endif()

if(morphizen_ENABLE_MLIR_BACKEND)
  add_compile_definitions(MORPHIZEN_ENABLE_MLIR_BACKEND=1)
else()
  add_compile_definitions(MORPHIZEN_ENABLE_MLIR_BACKEND=0)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/ort-bridge.cmake)
if(morphizen_ENABLE_UNIT_TEST AND morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
  add_subdirectory (test)
endif()
