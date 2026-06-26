##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# hip_custom_kernels - Custom HIP kernels for ops without ROCm library support
#
# Per-arch shared libraries: one SHARED library per arch in HIP_ARCHITECTURES,
# each carrying a single fatbin slice. The EP DLL dlopens the matching variant
# at JIT init via gcnArchName probing. No static archive is produced; the EP
# DLL no longer carries hip_* launcher symbols.
#
# Build flow (per arch):
#   .hip files -> hipcc -c --offload-arch=<arch> -> .obj
#                                               -> custom_kernels_<arch>.{dll,so}

cmake_minimum_required(VERSION 3.18)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip_utils.cmake)

# Require HIP_ARCHITECTURES to be set (no silent defaults that may not match hardware)
if(NOT HIP_ARCHITECTURES)
    message(FATAL_ERROR
        "[custom_kernels] HIP_ARCHITECTURES is not set.\n"
        "Pass -DHIP_ARCHITECTURES=<arch>[;<arch>...] to cmake (e.g. gfx1100;gfx1151).\n"
        "Run 'rocminfo | grep gfx' or 'hipInfo' to find your GPU architecture.")
endif()
message(STATUS "[custom_kernels] Using HIP_ARCHITECTURES: ${HIP_ARCHITECTURES}")

# Detect whether ANY architecture in HIP_ARCHITECTURES uses the RDNA 3 WMMA F16
# instruction in its legacy form. The Clang builtin
# `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` only matches the gfx11 variant
# (operand vectors <16 x half>); RDNA 4 (gfx12) implements the same operation
# with <8 x half> operands so the same builtin call fails ISel during the gfx12
# codegen pass with:
#   fatal error: error in backend: Cannot select: intrinsic
#       %llvm.amdgcn.wmma.f32.16x16x16.f16
# Detected arch families:
#   gfx1100 / gfx1101 / gfx1102 / gfx1103  — RDNA 3 dGPU (7900 series etc.)
#   gfx1150 / gfx1151                      — Strix Halo / Strix Point APU
# Anything else (gfx12xx RDNA 4, MI series CDNA, older gfx10xx, ...) gets
# HIPCK_RDNA4_WMMA=1 so matmul_nbits selects the half8 + `_gfx12` builtin
# variant. gemm_wmma_fp16 and the GQA fused-prefill WMMA kernel remain
# trap-stubs on non-RDNA-3 builds (no caller in lib/Runtime/ today).
# In a mixed multi-arch build (e.g. "gfx1151;gfx1201") even ONE RDNA 3 entry
# keeps HIPCK_RDNA3_WMMA — gfx12 device code still wouldn't ISel-select the
# legacy intrinsic, so multi-arch builds that include any gfx12 entry will FAIL
# at custom_kernels compile until per-arch object splitting lands.
set(_hipck_wmma_arch "")
foreach(_arch IN LISTS HIP_ARCHITECTURES)
    if(_arch MATCHES "^gfx110[0-3]$" OR _arch MATCHES "^gfx115[01]$")
        set(_hipck_wmma_arch "rdna3")
    endif()
endforeach()

set(_hipck_kernel_compile_opts "")
if(_hipck_wmma_arch STREQUAL "rdna3")
    message(STATUS
        "[custom_kernels] RDNA 3 arch detected in HIP_ARCHITECTURES "
        "(${HIP_ARCHITECTURES}) — HIPCK_RDNA3_WMMA=1.")
    list(APPEND _hipck_kernel_compile_opts -DHIPCK_RDNA3_WMMA=1)
else()
    message(STATUS
        "[custom_kernels] no RDNA 3 arch in HIP_ARCHITECTURES "
        "(${HIP_ARCHITECTURES}) — HIPCK_RDNA4_WMMA=1 (matmul_nbits WMMA "
        "uses the `_gfx12` builtin; gemm_wmma_fp16 and GQA fused-prefill "
        "WMMA remain trap-stubs).")
    list(APPEND _hipck_kernel_compile_opts -DHIPCK_RDNA4_WMMA=1)
endif()

set(_kernel_sources
    hip/rope_kernel.hip
    hip/gqa_kernel.hip
    hip/gemm_wmma_kernel.hip
    hip/elementwise_sub_kernel.hip
    hip/elementwise_reciprocal_kernel.hip
    hip/elementwise_sqrt_kernel.hip
    hip/elementwise_gelu_kernel.hip
    hip/leaky_relu_kernel.hip
    hip/elementwise_where_kernel.hip
    hip/elementwise_unary_kernel.hip
    hip/elementwise_binary_kernel.hip  # Mul/Add/Min/Max/Div/Mod/Equal/Less
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
    hip/pool_kernel.hip
    hip/resize_kernel.hip
    hip/global_pool_kernel.hip
    hip/matmul_nbits_kernel.hip
    hip/qmoe_kernel.hip
    hip/linear_attention_kernel.hip
    hip/transpose_kernel.hip
    hip/causal_conv_step_kernel.hip
    hip/slice_kernel.hip
    hip/scatter_nd_kernel.hip
    hip/gather_block_quantized_kernel.hip
    hip/softmax_kernel.hip
    hip/nonzero_kernel.hip
    hip/strided_copy_kernel.hip
)

# One SHARED target per arch. Each builds a fatbin with that single slice;
# OFFLOAD_ARCHS narrows the global list to one element so the resulting
# DLL/SO is not bloated with code for archs the consumer won't load.
foreach(_arch IN LISTS HIP_ARCHITECTURES)
    set(_target custom_kernels_${_arch})
    hip_add_library(${_target} SHARED ${_kernel_sources}
        INCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        COMPILE_OPTIONS
            ${_hipck_kernel_compile_opts}
        OFFLOAD_ARCHS
            ${_arch}
    )
    set_target_properties(${_target} PROPERTIES
        OUTPUT_NAME custom_kernels_${_arch}
        FOLDER "custom_kernels"
    )
    if(_hipck_wmma_arch STREQUAL "rdna3")
        target_compile_definitions(${_target} PUBLIC HIPCK_RDNA3_WMMA=1)
    else()
        target_compile_definitions(${_target} PUBLIC HIPCK_RDNA4_WMMA=1)
    endif()
    install(TARGETS ${_target}
        RUNTIME DESTINATION bin   # Windows .dll co-located with EP .dll
        LIBRARY DESTINATION lib   # Linux .so co-located with EP .so
        ARCHIVE DESTINATION lib   # Windows import lib; unused by the EP but harmless
    )
    message(STATUS "[custom_kernels] Adding per-arch SHARED target: ${_target}")
endforeach()

# Export include directory for Runtime bitcode compilation
set(HIP_CUSTOM_KERNELS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

# Header is intentionally NOT installed: the EP-side build always uses the
# in-tree header (3rd-party/custom_kernels/include/hip_custom_kernels.h) so
# the host-side declarations stay in lockstep with the kernel TUs that
# implement them.
