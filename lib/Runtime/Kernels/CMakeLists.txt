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
    hip/swish_kernel.hip
    hip/softplus_kernel.hip
    hip/elementwise_where_kernel.hip
    hip/qelementwise_kernel.hip
    hip/qactivation_kernel.hip
    hip/qmatmul_kernel.hip
    hip/qconv_kernel.hip
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

# MatMulNBits offline autotune table.
#
# The table is per-arch and so is this target, which is why the bytes are
# embedded here in custom_kernels_<arch>.dll (as is the GQA table below): a DLL
# carries exactly the table it can use.
#
# Only the checked-in lut/<arch>.fb ships; the linkable kMatmulNbitsLutData[]
# symbol is generated from it below with pure CMake (file(READ ... HEX)) -- the
# hip-ep build does not require Python, and no generated .cpp is committed (the
# compiled array is just the .fb's bytes). An arch with no measured table still
# builds via tools/empty_lut_data.cpp: the loader finds no table, and every
# shape falls through to the runtime sweep exactly as before. Shipping a new
# arch is therefore a table addition (a .fb), never a build break.
set(_matmul_nbits_autotune_dir ${CMAKE_CURRENT_SOURCE_DIR}/hip/autotune/matmul_nbits)

function(_matmul_nbits_lut_data_source arch out_var)
    set(_fb ${_matmul_nbits_autotune_dir}/lut/${arch}.fb)
    if(EXISTS ${_fb})
        # Re-run CMake configure if the table changes, so the generated source
        # below stays in sync with the checked-in .fb.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_fb})
        file(READ ${_fb} _hex HEX)
        string(LENGTH "${_hex}" _len)
        math(EXPR _n "${_len} / 2")
        # Each byte "nn" -> "0xnn,". The trailing comma before '}' is valid C++;
        # one long line is fine (compiled by hipcc/clang, no line-length limit).
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _body "${_hex}")
        set(_gen ${CMAKE_CURRENT_BINARY_DIR}/matmul_nbits_lut_data_${arch}.cpp)
        file(WRITE ${_gen}
"// Auto-generated from ${arch}.fb by lib/Runtime/Kernels/CMakeLists.txt; do not edit, do not commit.\n"
"#include <cstddef>\n"
"extern \"C\" const unsigned char kMatmulNbitsLutData[] = {${_body}};\n"
"extern \"C\" const size_t kMatmulNbitsLutData_size = ${_n};\n")
        set(${out_var} ${_gen} PARENT_SCOPE)
    else()
        message(STATUS
            "[custom_kernels] no MatMulNBits LUT for ${arch}; "
            "shapes will fall back to the runtime autotune")
        set(${out_var}
            ${_matmul_nbits_autotune_dir}/tools/empty_lut_data.cpp
            PARENT_SCOPE)
    endif()
endfunction()

# GQA offline autotune table + resolver.
#
# Both the table (lut/<arch>.fb) and the resolver (gqa_autotune.cpp) ride inside
# this per-arch DLL, next to gqa_kernel.hip. The resolver reads the table as an
# in-DLL data symbol; real/gqa.cpp (bitcode) reaches the resolver through the
# exported hip_gqa_autotune_* shims. This replaces the old runtime.bc embedding
# -- a data symbol does not cross the JIT boundary safely on Windows, so the
# table had to move in with its consumer.
#
# Only the checked-in lut/<arch>.fb ships; the linkable kGqaLutData[] symbol is
# generated from it below with pure CMake (file(READ ... HEX)) -- the hip-ep
# build needs no Python, and no generated .cpp is committed (the compiled array
# is just the .fb's bytes). Mirrors matmul_nbits. An arch with no measured table
# still builds: the size-0 stub makes the loader find no table and every shape
# falls to the heuristic. Shipping a new arch is a table addition, never a build
# break.
set(_gqa_autotune_dir ${CMAKE_CURRENT_SOURCE_DIR}/hip/autotune/gqa)

function(_gqa_lut_data_source arch out_var)
    set(_fb ${_gqa_autotune_dir}/lut/${arch}.fb)
    set(_gen ${CMAKE_CURRENT_BINARY_DIR}/gqa_lut_data_${arch}.cpp)
    # Arch-neutral symbol on purpose: each custom_kernels_<arch> DLL links
    # exactly one payload, so a fixed name lets gqa_autotune.cpp reference it
    # whatever arch it is built for.
    if(EXISTS ${_fb})
        # Re-run configure if the table changes, so the generated source stays
        # in sync with the checked-in .fb.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_fb})
        file(READ ${_fb} _hex HEX)
        string(LENGTH "${_hex}" _len)
        math(EXPR _n "${_len} / 2")
        # Each byte "nn" -> "0xnn,". The trailing comma before '}' is valid C++;
        # one long line is fine (compiled by hipcc/clang, no line-length limit).
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _body "${_hex}")
        file(WRITE ${_gen}
"// Auto-generated from ${arch}.fb by lib/Runtime/Kernels/CMakeLists.txt; do not edit, do not commit.\n"
"#include <cstddef>\n"
"extern \"C\" const unsigned char kGqaLutData[] = {${_body}};\n"
"extern \"C\" const size_t kGqaLutData_size = ${_n};\n")
    else()
        message(STATUS
            "[custom_kernels] no GQA LUT for ${arch}; "
            "shapes will use the compiled-in heuristic")
        file(WRITE ${_gen}
"// Auto-generated: no measured GQA LUT for ${arch}.\n"
"#include <cstddef>\n"
"extern \"C\" const unsigned char kGqaLutData[1] = {0};\n"
"extern \"C\" const size_t kGqaLutData_size = 0;\n")
    endif()
    set(${out_var} ${_gen} PARENT_SCOPE)
endfunction()

# One SHARED target per arch. Each builds a fatbin with that single slice;
# OFFLOAD_ARCHS narrows the global list to one element so the resulting
# DLL/SO is not bloated with code for archs the consumer won't load.
foreach(_arch IN LISTS HIP_ARCHITECTURES)
    set(_target custom_kernels_${_arch})
    _matmul_nbits_lut_data_source(${_arch} _matmul_lut_src)
    _gqa_lut_data_source(${_arch} _gqa_lut_src)
    hip_add_library(${_target} SHARED ${_kernel_sources}
            ${_matmul_nbits_autotune_dir}/matmul_nbits_autotune.cpp
            ${_matmul_lut_src}
            ${_gqa_autotune_dir}/gqa_autotune.cpp
            ${_gqa_lut_src}
        INCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            # matmul_nbits_autotune_generated.h and gqa_autotune_generated.h;
            # flatbuffers is header-only for reading, so no link library needed.
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

# The GQA offline-autotune LUT is embedded per-arch into custom_kernels_<arch>
# above (pure-CMake file(READ ... HEX) of lut/<arch>.fb), alongside its resolver
# gqa_autotune.cpp. No loose file is installed and no per-model/per-compile step
# is needed.

# Header is intentionally NOT installed: the EP-side build always uses the
# in-tree header (lib/Runtime/Kernels/include/hip_custom_kernels.h) so
# the host-side declarations stay in lockstep with the kernel TUs that
# implement them.
