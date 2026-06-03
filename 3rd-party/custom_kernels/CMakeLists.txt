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
    hip/gemm_wmma_kernel.hip
    hip/elementwise_sub_kernel.hip
    hip/elementwise_reciprocal_kernel.hip
    hip/elementwise_sqrt_kernel.hip
    hip/elementwise_gelu_kernel.hip
    hip/elementwise_where_kernel.hip
    hip/elementwise_unary_kernel.hip
    hip/elementwise_binary_kernel.hip
    hip/tile_kernel.hip
    hip/expand_kernel.hip
    hip/gather_nd_kernel.hip
    hip/cumsum_kernel.hip
    hip/pad_kernel.hip
    hip/layer_norm_kernel.hip
    hip/cast_kernel.hip
    hip/range_kernel.hip
    hip/gather_kernel.hip
    hip/reduce_sum_kernel.hip
    hip/global_pool_kernel.hip
    hip/pool_kernel.hip
    hip/resize_kernel.hip
    hip/matmul_nbits_kernel.hip
    hip/qmoe_kernel.hip
    hip/linear_attention_kernel.hip
    hip/transpose_kernel.hip
    hip/causal_conv_step_kernel.hip
    hip/slice_kernel.hip
    hip/scatter_nd_kernel.hip
    hip/elementwise_clip_kernel.hip
    hip/reduce_mean_kernel.hip
    INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Export the library path for CompilerDriver to discover at model-compile time
set(HIP_CUSTOM_KERNELS_LIB_PATH "$<TARGET_FILE:hip_custom_kernels>" PARENT_SCOPE)

# Export include directory for Runtime bitcode compilation
set(HIP_CUSTOM_KERNELS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

# Install the static library to the install prefix.
# The header is NOT installed to avoid version mismatch: EP builds should use
# the in-tree header at 3rd-party/custom_kernels/include/hip_custom_kernels.h
# to ensure the header always matches the compiled library.
install(TARGETS hip_custom_kernels
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

message(STATUS "[custom_kernels] Will install hip_custom_kernels.lib to \${CMAKE_INSTALL_PREFIX}/lib")
