##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# hip_custom_kernels - Custom HIP kernels for ops without ROCm library support
#
# This builds .hip files (containing __global__ kernels) into a static library
# using hipcc. The library is linked into model DLLs at model-compile time,
# alongside MIOpen/hipBLASLt/amdhip64 import libs.
#
# Build flow:
#   .hip files → hipcc -c → .obj → static lib (hip_custom_kernels.lib)
#   cmake --install → <prefix>/lib/hip_custom_kernels.lib
#                    → <prefix>/include/hip_custom_kernels.h

cmake_minimum_required(VERSION 3.18)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip_utils.cmake)

# Require HIP_ARCHITECTURES to be set (no silent defaults that may not match hardware)
if(NOT HIP_ARCHITECTURES)
    message(FATAL_ERROR
        "[custom_kernels] HIP_ARCHITECTURES is not set.\n"
        "Pass -DHIP_ARCHITECTURES=<arch> to cmake (e.g. gfx1100, gfx1151).\n"
        "Run 'rocminfo | grep gfx' or 'hipInfo' to find your GPU architecture.")
endif()
message(STATUS "[custom_kernels] Using HIP_ARCHITECTURES: ${HIP_ARCHITECTURES}")

# Build the static library from HIP sources
hip_add_library(hip_custom_kernels STATIC
    hip/rope_kernel.hip
    hip/gqa_kernel.hip
    hip/elementwise_sub_kernel.hip
    hip/cast_kernel.hip
    hip/gather_kernel.hip
    hip/reduce_sum_kernel.hip
    INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Export the library path for CompilerDriver to discover at model-compile time
set(hip_CUSTOM_KERNELS_LIB_PATH "$<TARGET_FILE:hip_custom_kernels>" PARENT_SCOPE)

# Export include directory for Runtime bitcode compilation
set(hip_CUSTOM_KERNELS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

# Install the static library and header to the install prefix.
# After `cmake --install`, CompilerDriver can find:
#   <prefix>/lib/hip_custom_kernels.lib  (static lib with compiled HIP kernels)
#   <prefix>/include/hip_custom_kernels.h (pure C header)
install(TARGETS hip_custom_kernels
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES include/hip_custom_kernels.h
    DESTINATION include
)

message(STATUS "[custom_kernels] Will install hip_custom_kernels.lib to \${CMAKE_INSTALL_PREFIX}/lib")
message(STATUS "[custom_kernels] Will install hip_custom_kernels.h to \${CMAKE_INSTALL_PREFIX}/include")
