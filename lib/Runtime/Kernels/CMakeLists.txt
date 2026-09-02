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

# Needed by the MatMulNBits LUT embedding below. lib/Runtime/CMakeLists.txt also
# looks for it, but that directory is traversed after this one, so the lookup
# has to happen here too -- find_package caches, so this is not a second search.
find_package(Python3 COMPONENTS Interpreter)

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

# MatMulNBits offline autotune table.
#
# The table is per-arch and so is this target, which is why the bytes are
# embedded here rather than in runtime.bc the way the GQA table is: a
# custom_kernels_<arch>.dll carries exactly the table it can use.
#
# An arch with no measured table still builds -- the stub below defines the
# symbol with size 0, the loader finds no table, and every shape falls through
# to the runtime sweep exactly as before. Shipping a new arch is therefore a
# table addition, never a build break.
set(_matmul_nbits_autotune_dir ${CMAKE_CURRENT_SOURCE_DIR}/hip/autotune/matmul_nbits)

function(_matmul_nbits_lut_data_source arch out_var)
    set(_fb ${_matmul_nbits_autotune_dir}/lut/${arch}.fb)
    set(_gen ${CMAKE_CURRENT_BINARY_DIR}/matmul_nbits_lut_data_${arch}.cpp)
    if(EXISTS ${_fb})
        add_custom_command(
            OUTPUT ${_gen}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/xxd.py
                    --var kMatmulNbitsLutData_${arch}
                    --output ${_gen}.tmp
                    ${_fb}
            # xxd.py emits a `static` array; rewrite it to the extern "C" symbol
            # matmul_nbits_autotune.cpp declares, and append the true size.
            COMMAND ${Python3_EXECUTABLE} -c
                    "import os,sys; p=sys.argv[1]; q=sys.argv[2]; n=os.path.getsize(sys.argv[3]); d=open(p).read().replace('static const unsigned char','extern \"C\" const unsigned char'); open(q,'w').write('#include <cstddef>\\n'+d+f'\\nextern \"C\" const size_t kMatmulNbitsLutData_${arch}_size = {n};\\n')"
                    ${_gen}.tmp ${_gen} ${_fb}
            COMMAND ${CMAKE_COMMAND} -E remove ${_gen}.tmp
            DEPENDS ${_fb} ${CMAKE_SOURCE_DIR}/cmake/xxd.py
            COMMENT "Embedding MatMulNBits LUT (${arch}.fb)"
            VERBATIM
        )
    else()
        message(STATUS
            "[custom_kernels] no MatMulNBits LUT for ${arch}; "
            "shapes will fall back to the runtime autotune")
        file(WRITE ${_gen}
"// Auto-generated: no measured MatMulNBits LUT for ${arch}.\n"
"#include <cstddef>\n"
"extern \"C\" const unsigned char kMatmulNbitsLutData_${arch}[1] = {0};\n"
"extern \"C\" const size_t kMatmulNbitsLutData_${arch}_size = 0;\n")
    endif()
    set(${out_var} ${_gen} PARENT_SCOPE)
endfunction()

# One SHARED target per arch. Each builds a fatbin with that single slice;
# OFFLOAD_ARCHS narrows the global list to one element so the resulting
# DLL/SO is not bloated with code for archs the consumer won't load.
foreach(_arch IN LISTS HIP_ARCHITECTURES)
    set(_target custom_kernels_${_arch})
    _matmul_nbits_lut_data_source(${_arch} _matmul_lut_src)
    hip_add_library(${_target} SHARED ${_kernel_sources}
            ${_matmul_nbits_autotune_dir}/matmul_nbits_autotune.cpp
            ${_matmul_lut_src}
        INCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            # matmul_nbits_autotune_generated.h; flatbuffers itself is
            # header-only for reading, so no link library is needed.
            ${CMAKE_BINARY_DIR}/schemas
            $<TARGET_PROPERTY:flatbuffers::flatbuffers,INTERFACE_INCLUDE_DIRECTORIES>
        OFFLOAD_ARCHS
            ${_arch}
        DEPENDS
            HipSchemas
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
