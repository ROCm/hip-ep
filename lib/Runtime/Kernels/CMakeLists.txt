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
    hip/softplus_kernel.hip
    hip/elementwise_where_kernel.hip
    hip/elementwise_unary_kernel.hip
    hip/elementwise_binary_kernel.hip  # Mul/Add/Min/Max/Div/Mod/Equal/Less
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
    hip/matmul_nbits_kernel.hip
    hip/qmoe_kernel.hip
    hip/qmoe_amd_kernel.hip
    hip/linear_attention_kernel.hip
    hip/transpose_kernel.hip
    hip/causal_conv_step_kernel.hip
    hip/slice_kernel.hip
    hip/scatter_nd_kernel.hip
    hip/gather_block_quantized_kernel.hip
    hip/qdq_kernel.hip
    hip/softmax_kernel.hip
    hip/nonzero_kernel.hip
    hip/strided_copy_kernel.hip
)

# hipRTC path (opt-in at runtime via HIPDNN_EP_RTC=1): the same device source
# that hipcc compiles below is also embedded as bytes and handed to hipRTC at
# runtime, so the two code objects are built from identical text.
find_package(hiprtc CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(_rtc_device_header
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/rtc/elementwise_sub_device.h)
set(_rtc_embedded_source
    ${CMAKE_CURRENT_BINARY_DIR}/elementwise_sub_device_source.cpp)

add_custom_command(
    OUTPUT ${_rtc_embedded_source}
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/cmake/xxd.py
            --var kElementwiseSubDeviceSource
            --output ${_rtc_embedded_source}.tmp
            ${_rtc_device_header}
    # xxd.py emits a `static` array with a trailing padding byte; rewrite it to
    # an extern "C" symbol and record the true byte count separately.
    COMMAND ${Python3_EXECUTABLE} -c
            "import os; data = open('${_rtc_embedded_source}.tmp').read(); data = data.replace('static const unsigned char', 'extern \"C\" const unsigned char'); n = os.path.getsize('${_rtc_device_header}'); data = '#include <cstddef>\\n' + data + f'\\nextern \"C\" const size_t kElementwiseSubDeviceSource_size = {n};\\n'; open('${_rtc_embedded_source}', 'w').write(data)"
    COMMAND ${CMAKE_COMMAND} -E remove ${_rtc_embedded_source}.tmp
    DEPENDS ${_rtc_device_header} ${CMAKE_SOURCE_DIR}/cmake/xxd.py
    COMMENT "Embedding elementwise_sub device source for hipRTC"
    VERBATIM
)

set(_rtc_sources
    rtc/hiprtc_module.hip
    ${_rtc_embedded_source}
)

# One SHARED target per arch. Each builds a fatbin with that single slice;
# OFFLOAD_ARCHS narrows the global list to one element so the resulting
# DLL/SO is not bloated with code for archs the consumer won't load.
foreach(_arch IN LISTS HIP_ARCHITECTURES)
    set(_target custom_kernels_${_arch})
    hip_add_library(${_target} SHARED ${_kernel_sources} ${_rtc_sources}
        INCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            hiprtc::hiprtc
        OFFLOAD_ARCHS
            ${_arch}
    )
    set_target_properties(${_target} PROPERTIES
        OUTPUT_NAME custom_kernels_${_arch}
        FOLDER "custom_kernels"
    )
    hipdnn_ep_apply_binary_compliance(${_target}
        DESCRIPTION "AMD GPU Execution Provider kernels (${_arch})")
    install(TARGETS ${_target}
        RUNTIME DESTINATION bin   # Windows .dll co-located with EP .dll
        LIBRARY DESTINATION lib   # Linux .so co-located with EP .so
        ARCHIVE DESTINATION lib   # Windows import lib; unused by the EP but harmless
    )
    message(STATUS "[custom_kernels] Adding per-arch SHARED target: ${_target}")
endforeach()

# Export include directory for Runtime bitcode compilation
set(HIP_CUSTOM_KERNELS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

# The GQA offline-autotune LUT is compiled into runtime.bc (see
# lib/Runtime/CMakeLists.txt, which xxd's lut/gfx1151.fb into the bitcode), so
# it travels with the build and is JIT-loaded with every model. No loose file
# is installed and no per-model/per-compile step is needed.

# Header is intentionally NOT installed: the EP-side build always uses the
# in-tree header (lib/Runtime/Kernels/include/hip_custom_kernels.h) so
# the host-side declarations stay in lockstep with the kernel TUs that
# implement them.
