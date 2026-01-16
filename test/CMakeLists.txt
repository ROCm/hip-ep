##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ROCm libraries configuration
set(THEROCK_DIST "" CACHE PATH "TheRock distribution path")

if(THEROCK_DIST)
  list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")
  list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")
endif()

find_package(hip QUIET)
find_package(miopen QUIET)
find_package(hipblaslt QUIET)

if(NOT hip_FOUND OR NOT miopen_FOUND OR NOT hipblaslt_FOUND)
  message(STATUS "ROCm libraries not found, skipping ROCm tests")
  return()
endif()

message(STATUS "Building ROCm tests")

# Conv test (MIOpen)
add_executable(rocm_conv_test test_conv.cpp)
target_link_libraries(rocm_conv_test PRIVATE MIOpen)
if(WIN32)
  target_include_directories(rocm_conv_test SYSTEM PRIVATE "${THEROCK_DIST}/include")
  target_compile_definitions(rocm_conv_test PRIVATE "__HIP_PLATFORM_AMD__=1")
  target_link_libraries(rocm_conv_test PRIVATE "${THEROCK_DIST}/lib/amdhip64.lib")
else()
  target_link_libraries(rocm_conv_test PRIVATE hip::device)
endif()
set_target_properties(rocm_conv_test PROPERTIES FOLDER "rocm-tests")

# Gemm test (hipBLASLt)
add_executable(rocm_gemm_test test_gemm.cpp)
target_link_libraries(rocm_gemm_test PRIVATE hipblaslt::hipblaslt)
if(WIN32)
  target_include_directories(rocm_gemm_test SYSTEM PRIVATE "${THEROCK_DIST}/include")
  target_compile_definitions(rocm_gemm_test PRIVATE "__HIP_PLATFORM_AMD__=1")
  target_link_libraries(rocm_gemm_test PRIVATE "${THEROCK_DIST}/lib/amdhip64.lib")
else()
  target_link_libraries(rocm_gemm_test PRIVATE hip::device)
endif()
set_target_properties(rocm_gemm_test PROPERTIES FOLDER "rocm-tests")
