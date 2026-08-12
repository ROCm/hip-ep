##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Custom HIP kernels for ops without ROCm library support.
#
# This tree lives under lib/Runtime/ because the kernels are the
# GPU-architecture-dependent half of the runtime (the EP/compiler proper are
# arch-independent). The build artifacts intentionally keep the
# custom_kernels_<arch> name regardless of the directory: the EP's JIT
# arch-probing (LlvmIrJit) and the CI co-location checks key on that exact
# filename, so the output names are decoupled from the directory name -- do not
# rename the targets to match the folder.
#
# Per-arch or per-family shared libraries driven by HIP_ARCHITECTURES:
#   * concrete arch (gfx1151): one SHARED lib, single native fatbin slice
#   * family name (gfx115X-all): one SHARED lib, multi-slice native fatbin
#   * semicolon list (gfx1150;gfx1151): one single-slice lib per entry
# LlvmIrJit dlopens custom_kernels_<gcnArchName> first, then the family DLL.
#
# Build flow (per arch):
#   .hip files -> hipcc -c --offload-arch=<arch> -> .obj
#                                               -> custom_kernels_<arch>.{dll,so}

cmake_minimum_required(VERSION 3.18)

include(${CMAKE_SOURCE_DIR}/cmake/hip_families.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip_utils.cmake)

# Require HIP_ARCHITECTURES to be set (no silent defaults that may not match hardware)
if(NOT HIP_ARCHITECTURES)
    message(FATAL_ERROR
        "[custom_kernels] HIP_ARCHITECTURES is not set.\n"
        "Pass -DHIP_ARCHITECTURES=<arch>[;<arch>...] or a family (e.g. gfx115X-all).\n"
        "Examples: gfx1151 (single arch), gfx115X-all (family fatbin), "
        "gfx1100;gfx1151 (multiple single-slice libs).\n"
        "Run 'rocminfo | grep gfx' or 'hipInfo' to find your GPU architecture.")
endif()
message(STATUS "[custom_kernels] Using HIP_ARCHITECTURES: ${HIP_ARCHITECTURES}")

set(_kernel_sources
    hip/rope_kernel.hip
    hip/gqa_kernel.hip
    hip/multi_head_attention_kernel.hip
    hip/gemm_wmma_kernel.hip
    hip/elementwise_sub_kernel.hip
    hip/elementwise_reciprocal_kernel.hip
    hip/elementwise_sqrt_kernel.hip
    hip/elementwise_gelu_kernel.hip
    hip/bias_gelu_kernel.hip
    hip/fast_gelu_kernel.hip
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
    hip/gather_elements_kernel.hip
    hip/top_k_kernel.hip
    hip/scatter_elements_kernel.hip
    hip/compress_kernel.hip
    hip/one_hot_kernel.hip
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

# One SHARED target per HIP_ARCHITECTURES entry. Families compile all member
# arches into one multi-slice fatbin; concrete entries get a single slice.
foreach(_entry IN LISTS HIP_ARCHITECTURES)
    set(_target custom_kernels_${_entry})
    hip_is_gpu_family("${_entry}" _is_family)
    if(_is_family)
        hip_expand_gpu_family("${_entry}" _offload_archs)
        if(NOT _offload_archs)
            message(FATAL_ERROR
                "[custom_kernels] family '${_entry}' has no member arches.")
        endif()
        message(STATUS
            "[custom_kernels] family '${_entry}' -> offload: ${_offload_archs}")
    else()
        set(_offload_archs ${_entry})
    endif()

    hip_add_library(${_target} SHARED ${_kernel_sources}
        INCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        OFFLOAD_ARCHS
            ${_offload_archs}
    )
    set_target_properties(${_target} PROPERTIES
        OUTPUT_NAME custom_kernels_${_entry}
        FOLDER "custom_kernels"
    )
    hipdnn_ep_apply_binary_compliance(${_target}
        DESCRIPTION "AMD GPU Execution Provider kernels (${_arch})")
    install(TARGETS ${_target}
        RUNTIME DESTINATION bin   # Windows .dll co-located with EP .dll
        LIBRARY DESTINATION lib   # Linux .so co-located with EP .so
        ARCHIVE DESTINATION lib   # Windows import lib; unused by the EP but harmless
    )
    if(_is_family)
        message(STATUS "[custom_kernels] Adding family SHARED target: ${_target}")
    else()
        message(STATUS "[custom_kernels] Adding per-arch SHARED target: ${_target}")
    endif()
endforeach()

# Export include directory for Runtime bitcode compilation
set(HIP_CUSTOM_KERNELS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

# Header is intentionally NOT installed: the EP-side build always uses the
# in-tree header (lib/Runtime/Kernels/include/hip_custom_kernels.h) so
# the host-side declarations stay in lockstep with the kernel TUs that
# implement them.
