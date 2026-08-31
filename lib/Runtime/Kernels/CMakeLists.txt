##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Custom HIP kernels for ops without ROCm library support.
#
# Split layout (family common + per-arch accel):
#   custom_kernels_gfx115x.{dll,so}  -- common .hip TUs, one RDNA3.5 device slice
#   custom_kernels_<arch>.{dll,so}   -- accel .hip TUs (WMMA/GQA/MatMulNBits/...)
#
# Older monolithic custom_kernels_<arch> targets are no longer produced.
#
# Build flow (per arch):
#   .hip files -> hipcc -c --offload-arch=<arch> -> .obj -> SHARED lib

cmake_minimum_required(VERSION 3.18)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip_utils.cmake)

if(NOT HIP_ARCHITECTURES)
    message(FATAL_ERROR
        "[custom_kernels] HIP_ARCHITECTURES is not set.\n"
        "Pass -DHIP_ARCHITECTURES=<arch>[;<arch>...] to cmake (e.g. gfx1100;gfx1151;gfx1152).\n"
        "Comma-separated lists are also accepted (gfx1150,gfx1151,gfx1152).\n"
        "Run 'rocminfo | grep gfx' or 'hipInfo' to find your GPU architecture.")
endif()
message(STATUS "[custom_kernels] Using HIP_ARCHITECTURES: ${HIP_ARCHITECTURES}")

# Common kernels: no WMMA / per-arch autotune megakernels (see docs/design/custom_kernel_family_split.md).
set(_kernel_sources_common
    hip/rope_kernel.hip
    hip/elementwise_sub_kernel.hip
    hip/elementwise_reciprocal_kernel.hip
    hip/elementwise_sqrt_kernel.hip
    hip/elementwise_gelu_kernel.hip
    hip/bias_gelu_kernel.hip
    hip/fast_gelu_kernel.hip
    hip/leaky_relu_kernel.hip
    hip/softplus_kernel.hip
    hip/elementwise_where_kernel.hip
    hip/elementwise_unary_kernel.hip
    hip/elementwise_binary_kernel.hip
    hip/tile_kernel.hip
    hip/expand_kernel.hip
    hip/gather_nd_kernel.hip
    hip/cumsum_kernel.hip
    hip/pad_kernel.hip
    hip/layer_norm_kernel.hip
    hip/instance_norm_kernel.hip
    hip/rms_norm_kernel.hip
    hip/skip_rms_norm_kernel.hip
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
    hip/conv_kernel.hip
    hip/conv_transpose_kernel.hip
    hip/resize_kernel.hip
    hip/grid_sample_kernel.hip
    hip/global_pool_kernel.hip
    hip/transpose_kernel.hip
    hip/causal_conv_step_kernel.hip
    hip/slice_kernel.hip
    hip/scatter_nd_kernel.hip
    hip/gather_block_quantized_kernel.hip
    hip/softmax_kernel.hip
    hip/nonzero_kernel.hip
    hip/strided_copy_kernel.hip
)

# Accel kernels: WMMA/GQA/MatMulNBits/QMoE/attention megakernels + autotune.
set(_kernel_sources_accel
    hip/gqa_kernel.hip
    hip/multi_head_attention_kernel.hip
    hip/gemm_wmma_kernel.hip
    hip/matmul_nbits_kernel.hip
    hip/qmoe_kernel.hip
    hip/linear_attention_kernel.hip
)

set(_CK_FAMILY_GFX115X_MEMBERS gfx1150 gfx1151 gfx1152 gfx1153)
set(_CK_NEEDS_GFX115X FALSE)
foreach(_arch IN LISTS HIP_ARCHITECTURES)
    if(_arch IN_LIST _CK_FAMILY_GFX115X_MEMBERS)
        set(_CK_NEEDS_GFX115X TRUE)
    endif()
endforeach()

function(_hip_ep_install_custom_kernels_target _target _desc)
    hipdnn_ep_apply_binary_compliance(${_target} DESCRIPTION "${_desc}")
    install(TARGETS ${_target}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endfunction()

# RDNA 3.5 family common library (device slice: gfx1151 representative).
if(_CK_NEEDS_GFX115X)
    set(_target custom_kernels_gfx115x)
    hip_add_library(${_target} SHARED ${_kernel_sources_common}
        INCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        OFFLOAD_ARCHS
            gfx1151
    )
    set_target_properties(${_target} PROPERTIES
        OUTPUT_NAME custom_kernels_gfx115x
        FOLDER "custom_kernels"
    )
    _hip_ep_install_custom_kernels_target(${_target}
        "AMD GPU EP common kernels (RDNA3.5 family gfx115x)")
    message(STATUS "[custom_kernels] Adding family common SHARED target: ${_target}")
endif()

# Per-arch common for GPUs outside the gfx115x family.
foreach(_arch IN LISTS HIP_ARCHITECTURES)
    if(_arch IN_LIST _CK_FAMILY_GFX115X_MEMBERS)
        continue()
    endif()
    set(_target custom_kernels_${_arch}_common)
    hip_add_library(${_target} SHARED ${_kernel_sources_common}
        INCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        OFFLOAD_ARCHS
            ${_arch}
    )
    set_target_properties(${_target} PROPERTIES
        OUTPUT_NAME custom_kernels_${_arch}_common
        FOLDER "custom_kernels"
    )
    _hip_ep_install_custom_kernels_target(${_target}
        "AMD GPU EP common kernels (${_arch})")
    message(STATUS "[custom_kernels] Adding per-arch common SHARED target: ${_target}")
endforeach()

# Per-arch accel libraries.
foreach(_arch IN LISTS HIP_ARCHITECTURES)
    set(_target custom_kernels_${_arch})
    hip_add_library(${_target} SHARED ${_kernel_sources_accel}
        INCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        OFFLOAD_ARCHS
            ${_arch}
    )
    set_target_properties(${_target} PROPERTIES
        OUTPUT_NAME custom_kernels_${_arch}
        FOLDER "custom_kernels"
    )
    _hip_ep_install_custom_kernels_target(${_target}
        "AMD GPU EP accel kernels (${_arch})")
    message(STATUS "[custom_kernels] Adding per-arch accel SHARED target: ${_target}")
endforeach()

set(HIP_CUSTOM_KERNELS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

# The GQA offline-autotune LUT is compiled into runtime.bc (see
# lib/Runtime/CMakeLists.txt, which xxd's lut/gfx1151.fb into the bitcode), so
# it travels with the build and is JIT-loaded with every model. No loose file
# is installed and no per-model/per-compile step is needed.

# Header is intentionally NOT installed: the EP-side build always uses the
# in-tree header (lib/Runtime/Kernels/include/hip_custom_kernels.h) so
# the host-side declarations stay in lockstep with the kernel TUs that
# implement them.
