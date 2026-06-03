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

# Detect whether ANY architecture in HIP_ARCHITECTURES supports the RDNA 3
# WMMA F16 instruction in its legacy form. The Clang builtin
# `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` only matches the gfx11
# variant (operand vectors <16 x half>); RDNA 4 (gfx12) implements the
# same operation with <8 x half> operands so the same builtin call fails
# ISel during the gfx12 codegen pass with:
#   fatal error: error in backend: Cannot select: intrinsic
#       %llvm.amdgcn.wmma.f32.16x16x16.f16
# Detected arch families:
#   gfx1100 / gfx1101 / gfx1102 / gfx1103  — RDNA 3 dGPU (7900 series etc.)
#   gfx1150 / gfx1151                      — Strix Halo / Strix Point APU
# Anything else (gfx12xx RDNA 4, MI series CDNA, older gfx10xx, ...) gets
# HIPCK_NO_RDNA3_WMMA=1 propagated to every WMMA-using kernel + dispatcher
# so the build skips the broken codegen and routes to non-WMMA fallbacks
# at runtime. In a mixed multi-arch build (e.g. "gfx1151;gfx1201") even ONE
# RDNA 3 entry keeps the macro undefined and the kernels compile for every
# listed arch — gfx12 device code still wouldn't ISel-select the broken
# intrinsic, so multi-arch builds that include any gfx12 entry will FAIL
# at custom_kernels compile until Option B (true gfx12 WMMA port) lands.
set(_hipck_has_rdna3_wmma 0)
foreach(_arch IN LISTS HIP_ARCHITECTURES)
    if(_arch MATCHES "^gfx110[0-3]$" OR _arch MATCHES "^gfx115[01]$")
        set(_hipck_has_rdna3_wmma 1)
    endif()
endforeach()

# Compose the compile flags BEFORE hip_add_library: on Windows the macro
# defined by hip_utils' `_hip_compile_sources` invokes hipcc via
# add_custom_command(), which only reads the flags listed in COMPILE_OPTIONS
# (NOT target_compile_definitions added afterwards on the cmake target).
# On Linux native HIP language support would honor target_compile_definitions
# too, but we route everything through COMPILE_OPTIONS for parity.
set(_hipck_kernel_compile_opts "")
if(NOT _hipck_has_rdna3_wmma)
    message(STATUS
        "[custom_kernels] no RDNA 3 arch in HIP_ARCHITECTURES "
        "(${HIP_ARCHITECTURES}) — disabling WMMA F16 path "
        "(HIPCK_NO_RDNA3_WMMA=1). gemm_wmma_fp16 and the GQA fused "
        "prefill WMMA kernel become trap-stubs; matmul_nbits routes "
        "every shape through GEMV. Expect lower prefill TPS until the "
        "gfx12 WMMA port lands.")
    list(APPEND _hipck_kernel_compile_opts -DHIPCK_NO_RDNA3_WMMA=1)
endif()

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
    hip/matmul_nbits_kernel.hip
    hip/qmoe_kernel.hip
    hip/linear_attention_kernel.hip
    hip/transpose_kernel.hip
    hip/causal_conv_step_kernel.hip
    hip/slice_kernel.hip
    hip/scatter_nd_kernel.hip
    INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    COMPILE_OPTIONS
        ${_hipck_kernel_compile_opts}
)

# Belt-and-suspenders for downstream C++ consumers (runtime EP, model.dll
# bitcode TUs) that #include hip_custom_kernels.h directly via the
# in-tree path. The header itself doesn't branch on HIPCK_NO_RDNA3_WMMA
# today, but a future host-side dispatcher in the EP could — pin the
# macro PUBLIC on the cmake target so the kernel TU and the EP TUs
# can never disagree.
if(NOT _hipck_has_rdna3_wmma)
    target_compile_definitions(hip_custom_kernels PUBLIC HIPCK_NO_RDNA3_WMMA=1)
endif()

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
