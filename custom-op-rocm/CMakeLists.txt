##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.17)
project(
  morphizen-custom-op-rocm
  VERSION 1.0.0
  LANGUAGES C CXX)

find_package(glog CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

# ROCm libraries configuration
set(THEROCK_DIST "" CACHE PATH "TheRock distribution path")

if(THEROCK_DIST)
  list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")
  list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")
endif()

find_package(hip QUIET)
find_package(miopen QUIET)
find_package(hipblaslt QUIET)

if(hip_FOUND)
  message(STATUS "Found HIP: ${hip_DIR}")
else()
  message(FATAL_ERROR "HIP is required but not found. Please install TheRock ROCm SDK.")
endif()

if(miopen_FOUND)
  message(STATUS "Found MIOpen: ${miopen_DIR}")
else()
  message(FATAL_ERROR "MIOpen is required but not found. Please install TheRock ROCm SDK.")
endif()

if(hipblaslt_FOUND)
  message(STATUS "Found hipBLASLt: ${hipblaslt_DIR}")
else()
  message(FATAL_ERROR "hipBLASLt is required but not found. Please install TheRock ROCm SDK.")
endif()

set(LIB_NAME morphizen-custom-op-rocm)
add_library(${LIB_NAME} STATIC src/main.cpp src/custom_op.hpp src/custom_op.cpp)
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "rocm")
target_include_directories(${LIB_NAME} PRIVATE include)
target_compile_definitions(${LIB_NAME} PRIVATE "VAIP_CUSTOM_OP=1")

# Link ROCm libraries
target_link_libraries(${LIB_NAME} PRIVATE MIOpen)
target_link_libraries(${LIB_NAME} PRIVATE hipblaslt::hipblaslt)

if(WIN32)
  # On Windows with MSVC, add HIP include directories manually
  target_include_directories(${LIB_NAME} SYSTEM PRIVATE "${THEROCK_DIST}/include")
  target_compile_definitions(${LIB_NAME} PRIVATE "__HIP_PLATFORM_AMD__=1")
  # Link the HIP runtime library (amdhip64.dll)
  target_link_libraries(${LIB_NAME} PRIVATE "${THEROCK_DIST}/lib/amdhip64.lib")
else()
  # On Linux, use hip::device normally
  target_link_libraries(${LIB_NAME} PRIVATE hip::device)
endif()

target_link_libraries(${LIB_NAME} PRIVATE
  rocm_proto
  glog::glog
  morphizen::morphizen-core-static
)
set_target_properties(${LIB_NAME} PROPERTIES
  OUTPUT_NAME "${LIB_NAME}")
target_link_libraries(${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
