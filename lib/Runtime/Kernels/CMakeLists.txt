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

# Composable Kernel (bundled in the TheRock ROCm SDK) backs the Conv forward op
# (hip/conv_kernel.cpp). CK's package config only defines a component's imported
# target when that component is requested, so ask for device_conv_operations
# explicitly. When unavailable, Conv stays on the MIOpen path in
# lib/Runtime/real/miopen.cpp (the wrapper guards the CK call on the compile
# define below).
set(_ck_conv_link_libs "")
set(_ck_conv_includes "")
if(NOT DEFINED THEROCK_DIST AND DEFINED ENV{THEROCK_DIST})
    set(THEROCK_DIST "$ENV{THEROCK_DIST}")
endif()
if(THEROCK_DIST)
    list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}" "${THEROCK_DIST}/lib/cmake")
endif()
# CK's imported targets carry a Threads::Threads link dependency, so the
# Threads package must be resolved before its config is included.
find_package(Threads QUIET)
find_package(composable_kernel CONFIG QUIET COMPONENTS device_conv_operations)
if(composable_kernel_FOUND AND TARGET composable_kernel::device_conv_operations)
    message(STATUS "[custom_kernels] Composable Kernel found -- building CK Conv forward")
    # Link the prebuilt CK static library by FILE PATH rather than via the
    # imported target. The target's INTERFACE_LINK_LIBRARIES pulls in
    # hip::device, whose `-x hip` INTERFACE_COMPILE_OPTION would leak onto the
    # MSVC-compiled dummy TU (LINK_ONLY does not wrap transitive link deps).
    # conv_kernel.cpp is compiled by hipcc via a custom command that gets the CK
    # headers through the explicit INCLUDE_DIRECTORIES path below, so the target
    # usage-requirements are not needed here.
    get_target_property(_ck_conv_lib_file
        composable_kernel::device_conv_operations IMPORTED_LOCATION_RELEASE)
    if(NOT _ck_conv_lib_file)
        get_target_property(_ck_conv_lib_file
            composable_kernel::device_conv_operations LOCATION)
    endif()
    set(_ck_conv_link_libs "${_ck_conv_lib_file}")
    get_target_property(_ck_conv_includes
        composable_kernel::device_conv_operations INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _ck_conv_includes)
        set(_ck_conv_includes "${THEROCK_DIST}/include")
    endif()
    # Propagate to the top-level scope so lib/Runtime's bitcode compile of
    # miopen.cpp enables the CK dispatch branch (see lib/Runtime/CMakeLists.txt).
    set(HIPDNN_EP_USE_CK_CONV ON CACHE INTERNAL "CK-backed Conv forward available")
else()
    message(STATUS "[custom_kernels] Composable Kernel not found -- Conv stays on MIOpen")
    set(HIPDNN_EP_USE_CK_CONV OFF CACHE INTERNAL "CK-backed Conv forward available")
endif()

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
# conv_kernel.cpp needs the CK device_conv_operations library; only build it
# when CK was found above.
if(_ck_conv_link_libs)
    list(APPEND _kernel_sources hip/conv_kernel.cpp)
    # CK headers need host+device constexpr; exclude this TU from the MSVC
    # cmath `-fno-cuda-host-device-constexpr` workaround (see hip_utils.cmake).
    set(HIP_UTILS_NO_CONSTEXPR_WORKAROUND_SOURCES conv_kernel)
endif()

# CK's prebuilt static lib is compiled against the dynamic CRT (/MD). Linking
# it into this DLL under the project-default static CRT (/MT) produces LNK1169
# (both libcmt and msvcrt CRTs). custom_kernels_<arch> is a standalone DLL the
# EP dlopens and talks to over a pure C ABI (no CRT objects cross the boundary),
# so switching just this DLL to the dynamic CRT to match CK is safe.
if(_ck_conv_link_libs AND WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    message(STATUS "[custom_kernels] Using dynamic CRT (/MD) to match Composable Kernel")
endif()

# One SHARED target per arch. Each builds a fatbin with that single slice;
# OFFLOAD_ARCHS narrows the global list to one element so the resulting
# DLL/SO is not bloated with code for archs the consumer won't load.
foreach(_arch IN LISTS HIP_ARCHITECTURES)
    set(_target custom_kernels_${_arch})
    hip_add_library(${_target} SHARED ${_kernel_sources}
        INCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${_ck_conv_includes}
        # Composable Kernel headers require C++17 (std::optional, std::byte,
        # template<auto>); the other kernels already rely on C++17 features via
        # -Wc++17-extensions, so promoting the whole library is safe.
        COMPILE_OPTIONS
            -std=c++17
        LINK_LIBRARIES
            ${_ck_conv_link_libs}
        OFFLOAD_ARCHS
            ${_arch}
    )
    set_target_properties(${_target} PROPERTIES
        OUTPUT_NAME custom_kernels_${_arch}
        FOLDER "custom_kernels"
    )
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
# in-tree header (lib/Runtime/Kernels/include/hip_custom_kernels.h) so
# the host-side declarations stay in lockstep with the kernel TUs that
# implement them.
