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
    hip/qmatmul_kernel.hip
    hip/qgemm_kernel.hip
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
# The tables are embedded here in custom_kernels_<arch>.dll (as is GQA below).
# A gfx*-generic DLL carries every family member's table that exists on disk
# (see _lut_arch_members); at load compatible() keeps the one whose gpu_arch
# matches the device and ignores the rest, so each member ISA gets its own table
# and the others fall back to runtime autotune -- no cross-arch mixing.
#
# Only the checked-in lut/<arch>.fb files ship; the linkable blob registry
# (kMatmulNbitsLutBlobs[]) is generated from them below with pure CMake
# (file(READ ... HEX)) -- the hip-ep build does not require Python, and no
# generated .cpp is committed (the arrays are just the .fb bytes). An arch with
# no measured table still builds -- _emit_lut_registry emits an empty registry
# (BlobCount 0) -- so the loader finds no table and every shape falls through to
# the runtime sweep. Shipping a new arch is a table addition (a .fb), never a
# build break. (tools/empty_lut_data.cpp is only for the manual sweep driver.)
set(_matmul_nbits_autotune_dir ${CMAKE_CURRENT_SOURCE_DIR}/hip/autotune/matmul_nbits)

# Maps a build arch to the concrete family members whose LUT may ship in its
# DLL. Mirrors genericTargetFor() in backend-mlir-compiler/custom-op-mlir/src/
# LlvmIrJit.cpp: a gfx*-generic DLL runs on every member ISA, so it carries each
# member's table that exists on disk and the loader picks the one matching the
# live device by gpu_arch. A concrete arch maps to itself.
function(_lut_arch_members arch out_var)
    if(arch STREQUAL "gfx9-generic")
        set(_m gfx900 gfx902 gfx904 gfx906 gfx909 gfx90c)
    elseif(arch STREQUAL "gfx9-4-generic")
        set(_m gfx942 gfx950)
    elseif(arch STREQUAL "gfx10-1-generic")
        set(_m gfx1010 gfx1011 gfx1012 gfx1013)
    elseif(arch STREQUAL "gfx10-3-generic")
        set(_m gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036)
    elseif(arch STREQUAL "gfx11-generic")
        set(_m gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153)
    elseif(arch STREQUAL "gfx12-generic")
        set(_m gfx1200 gfx1201)
    else()
        set(_m ${arch})
    endif()
    set(${out_var} ${_m} PARENT_SCOPE)
endfunction()

# Generates a .cpp embedding every existing lut/<member>.fb for `arch` as a blob
# registry: <sym>Blobs[] / <sym>BlobSizes[] / <sym>BlobCount. The loader loads
# the first blob whose gpu_arch matches the device (compatible() rejects the
# rest), so a single gfx*-generic DLL serves each member arch its own table with
# no cross-arch mixing. Members with no lut/<member>.fb are skipped; an arch that
# matches none yields an empty (count 0) registry and every shape autotunes --
# never a build error.
function(_emit_lut_registry arch lut_dir sym out_var)
    _lut_arch_members(${arch} _members)
    set(_arrays "")
    set(_ptrs "")
    set(_sizes "")
    set(_count 0)
    foreach(_m IN LISTS _members)
        set(_fb ${lut_dir}/lut/${_m}.fb)
        if(EXISTS ${_fb})
            # Re-run configure if a table changes, keeping the generated source
            # in sync with the checked-in .fb.
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_fb})
            file(READ ${_fb} _hex HEX)
            string(LENGTH "${_hex}" _len)
            math(EXPR _n "${_len} / 2")
            string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _body "${_hex}")
            string(APPEND _arrays "static const unsigned char _blob_${_m}[] = {${_body}};\n")
            string(APPEND _ptrs "  _blob_${_m},\n")
            string(APPEND _sizes "  ${_n},\n")
            math(EXPR _count "${_count} + 1")
        endif()
    endforeach()
    set(_gen ${CMAKE_CURRENT_BINARY_DIR}/${sym}_data_${arch}.cpp)
    if(_count EQUAL 0)
        message(STATUS
            "[custom_kernels] no ${sym} LUT for ${arch}; "
            "shapes will fall back to the runtime autotune")
        file(WRITE ${_gen}
"// Auto-generated by lib/Runtime/Kernels/CMakeLists.txt; do not edit, do not commit.\n"
"#include <cstddef>\n"
"extern \"C\" const unsigned char* const ${sym}Blobs[1] = {nullptr};\n"
"extern \"C\" const size_t ${sym}BlobSizes[1] = {0};\n"
"extern \"C\" const size_t ${sym}BlobCount = 0;\n")
    else()
        file(WRITE ${_gen}
"// Auto-generated by lib/Runtime/Kernels/CMakeLists.txt; do not edit, do not commit.\n"
"#include <cstddef>\n"
"${_arrays}"
"extern \"C\" const unsigned char* const ${sym}Blobs[] = {\n${_ptrs}};\n"
"extern \"C\" const size_t ${sym}BlobSizes[] = {\n${_sizes}};\n"
"extern \"C\" const size_t ${sym}BlobCount = ${_count};\n")
    endif()
    set(${out_var} ${_gen} PARENT_SCOPE)
endfunction()

function(_matmul_nbits_lut_data_source arch out_var)
    _emit_lut_registry(${arch} ${_matmul_nbits_autotune_dir} kMatmulNbitsLut _gen)
    set(${out_var} ${_gen} PARENT_SCOPE)
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
# Only the checked-in lut/<arch>.fb files ship; the linkable blob registry
# (kGqaLutBlobs[]) is generated from them below with pure CMake (file(READ ...
# HEX)) -- the hip-ep build needs no Python, and no generated .cpp is committed
# (the arrays are just the .fb bytes). Mirrors matmul_nbits: a gfx*-generic DLL
# embeds every family member's table, and compatible() picks the device's at
# load. An arch with no measured table yields an empty registry, so the loader
# finds no table and every shape falls to the heuristic. Shipping a new arch is
# a table addition, never a build break.
set(_gqa_autotune_dir ${CMAKE_CURRENT_SOURCE_DIR}/hip/autotune/gqa)

function(_gqa_lut_data_source arch out_var)
    _emit_lut_registry(${arch} ${_gqa_autotune_dir} kGqaLut _gen)
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
