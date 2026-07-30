##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# HipDnnEpRuntime - Runtime library for ONNX HIP DNN Execution Provider
# Provides HIP/MIOpen/hipBLASLt wrappers called by generated LLVM IR
#
# This implements a two-stage compilation:
# Stage 1 (Build Time): runtime.cpp → bitcode → embedded in library
# Stage 2 (Model Compilation): Generated MLIR + runtime.bc → optimized DLL

cmake_minimum_required(VERSION 3.18)

# ============================================================================
# STEP 1: Find Required Tools
# ============================================================================

# Python for xxd.py (embeds binary data as C array)
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_FOUND)
    # Try just python on Windows
    find_program(PYTHON_EXECUTABLE python)
    if(NOT PYTHON_EXECUTABLE)
        message(FATAL_ERROR "Python not found. Required for xxd.py script.")
    endif()
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

message(STATUS "Found Python: ${Python3_EXECUTABLE}")

# Clang REQUIRED for bitcode generation.
# When LLVM is built from source as a subproject the `clang` target exists but
# its .exe is not present at configure time, so find_program cannot see it --
# use the target directly (the bitcode custom commands below then gain a
# build-time dependency on it). Fall back to find_program for the prebuilt /
# installed-LLVM case.
if(TARGET clang)
    set(CLANG_EXECUTABLE clang)
    message(STATUS "Using in-tree clang target for bitcode generation")
else()
    find_program(CLANG_EXECUTABLE NAMES clang clang-cl
        PATHS
            ${LLVM_BINARY_DIR}/bin       # When LLVM built from source
            ${LLVM_TOOLS_BINARY_DIR}     # When LLVM installed via package
    )
endif()

if(NOT CLANG_EXECUTABLE)
    message(FATAL_ERROR
        "Clang compiler not found. Runtime IR merging is REQUIRED.\n"
        "\n"
        "Why Clang is needed:\n"
        "  - Compiles Runtime to LLVM bitcode (.bc) for zero-cost abstraction\n"
        "  - Enables cross-module inlining (generated code + Runtime)\n"
        "  - MSVC and GCC cannot produce compatible LLVM bitcode\n"
        "\n"
        "To fix:\n"
        "  - Install LLVM with Clang: https://releases.llvm.org/\n"
        "  - Or build LLVM from source with -DLLVM_ENABLE_PROJECTS=clang\n"
        "\n"
        "Searched in:\n"
        "  - ${LLVM_BINARY_DIR}/bin\n"
        "  - ${LLVM_TOOLS_BINARY_DIR}\n"
        "  - System PATH\n")
endif()

message(STATUS "Found Clang for bitcode generation: ${CLANG_EXECUTABLE}")

# llvm-link REQUIRED for combining bitcode files. Same in-tree-target-first
# logic as clang above (llvm-link is a standard LLVM tool target).
if(TARGET llvm-link)
    set(LLVM_LINK_EXECUTABLE llvm-link)
    message(STATUS "Using in-tree llvm-link target")
else()
    find_program(LLVM_LINK_EXECUTABLE NAMES llvm-link
        PATHS
            ${LLVM_BINARY_DIR}/bin
            ${LLVM_TOOLS_BINARY_DIR}
    )
endif()

if(NOT LLVM_LINK_EXECUTABLE)
    message(FATAL_ERROR "llvm-link not found (required for combining runtime modules)")
endif()

message(STATUS "Found llvm-link: ${LLVM_LINK_EXECUTABLE}")

# ============================================================================
# STEP 1b: Find morphizen-foundation headers
# ============================================================================

find_path(MORPHIZEN_FOUNDATION_INCLUDE_DIR
    NAMES morphizen-foundation/file_io.hpp
    PATHS ${CMAKE_SOURCE_DIR}/include ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)
if(NOT MORPHIZEN_FOUNDATION_INCLUDE_DIR)
    message(FATAL_ERROR
        "morphizen-foundation/file_io.hpp not found.\n"
        "Expected in-tree at include/morphizen-foundation/file_io.hpp.")
endif()
message(STATUS "Found morphizen-foundation: ${MORPHIZEN_FOUNDATION_INCLUDE_DIR}")

# ============================================================================
# STEP 2: Determine Runtime Mode (Mock or Real)
# ============================================================================
# BUILD_MOCK_RUNTIME is declared once in the top-level CMakeLists.txt.

if(NOT BUILD_MOCK_RUNTIME)
    # Try to find ROCm components
    # Accept THEROCK_DIST from either CMake variable or environment variable
    if(NOT DEFINED THEROCK_DIST AND DEFINED ENV{THEROCK_DIST})
        set(THEROCK_DIST "$ENV{THEROCK_DIST}")
    endif()

    if(DEFINED THEROCK_DIST)
        message(STATUS "Using TheRock ROCm SDK at: ${THEROCK_DIST}")

        set(ORIGINAL_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
        list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")
        list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")
    endif()

    find_package(hip QUIET)
    if(NOT HIPDNN_EP_DISABLE_VENDOR_BLAS)
        find_package(MIOpen QUIET)
        find_package(hipblaslt QUIET)
    endif()

    if(DEFINED THEROCK_DIST)
        set(CMAKE_PREFIX_PATH "${ORIGINAL_CMAKE_PREFIX_PATH}")
    endif()

    # Vendor BLAS/DNN disabled: build the real runtime with only hip (the stubbed
    # wrappers, added in the follow-up change, keep the ABI intact).
    if(HIPDNN_EP_DISABLE_VENDOR_BLAS)
        if(NOT hip_FOUND)
            message(WARNING "HIP not found. Building mock runtime library for testing.")
            message(WARNING "  hip_FOUND=${hip_FOUND}; set THEROCK_DIST to the ROCm SDK path")
            set(BUILD_MOCK_RUNTIME TRUE)
        else()
            message(STATUS "Building REAL runtime with HIP (vendor MIOpen/hipBLASLt DISABLED)")
            message(STATUS "  HIP: ${hip_DIR}")
        endif()
    elseif(NOT hip_FOUND OR NOT MIOpen_FOUND OR NOT hipblaslt_FOUND)
        message(WARNING "ROCm components not found. Building mock runtime library for testing.")
        message(WARNING "  hip_FOUND=${hip_FOUND}, MIOpen_FOUND=${MIOpen_FOUND}, hipblaslt_FOUND=${hipblaslt_FOUND}")
        message(WARNING "  Set THEROCK_DIST environment variable to ROCm SDK path")
        set(BUILD_MOCK_RUNTIME TRUE)
    else()
        message(STATUS "Building REAL runtime with HIP/MIOpen/hipBLASLt")
        message(STATUS "  HIP: ${hip_DIR}")
        message(STATUS "  MIOpen: ${MIOpen_DIR}")
        message(STATUS "  hipBLASLt: ${hipblaslt_DIR}")
    endif()
else()
    message(STATUS "Building mock runtime (BUILD_MOCK_RUNTIME explicitly set)")
endif()

# ============================================================================
# STEP 3: Generate LLVM Bitcode from Runtime Modules
# ============================================================================

# Macro to compile a single C++ file to bitcode
macro(compile_to_bitcode SOURCE_FILE OUTPUT_BC)
    # Build include flags
    # NOTE: -I and path are separate list items to handle paths with spaces
    # (e.g., "C:/Program Files/..."). Clang supports "-I" "<path>" as two args.
    set(COMPILE_FLAGS "")

    # PIC is required when this bitcode is later linked into the per-model
    # shared library on Linux (ELF .so). Without -fPIC the IR omits dso_local
    # markers, codegen emits R_X86_64_PC32 against statics like the Itanium
    # guard variables for function-local statics, and lld rejects them with
    # "relocation cannot be used against symbol; recompile with -fPIC". On
    # Windows COFF this flag is a no-op (clang ignores it for the MSVC ABI).
    if(NOT WIN32)
        list(APPEND COMPILE_FLAGS "-fPIC")
    endif()

    # Propagate the vendor-BLAS opt-out into the bitcode compile so the
    # MIOpen/hipBLASLt-backed wrappers (real/{miopen,hipblas,elementwise,gemm}.cpp)
    # compile their stub branch instead of the vendor-backed one.
    if(HIPDNN_EP_DISABLE_VENDOR_BLAS)
        list(APPEND COMPILE_FLAGS "-DHIPDNN_EP_DISABLE_VENDOR_BLAS")
    endif()

    # Always add root directory for public headers
    list(APPEND COMPILE_FLAGS "-I" "${CMAKE_CURRENT_SOURCE_DIR}")

    # Shared project headers (timing.h, etc.)
    list(APPEND COMPILE_FLAGS "-I" "${CMAKE_SOURCE_DIR}/include")

    # Add morphizen-foundation headers
    list(APPEND COMPILE_FLAGS "-I" "${MORPHIZEN_FOUNDATION_INCLUDE_DIR}")

    # Add schemas generated headers (model_metadata_generated.h etc.)
    list(APPEND COMPILE_FLAGS "-I" "${CMAKE_BINARY_DIR}/schemas")

    # Add flatbuffers headers
    get_target_property(_fb_incs flatbuffers::flatbuffers INTERFACE_INCLUDE_DIRECTORIES)
    if(_fb_incs)
        foreach(_inc ${_fb_incs})
            list(APPEND COMPILE_FLAGS "-I" "${_inc}")
        endforeach()
    endif()

    # Add implementation directory and collect impl-specific deps. Both
    # `real/` and `mock/` ship their own `runtime_types.h` (resolved via the
    # -I flag below), so the DEPENDS list must reference whichever copy the
    # active build will actually compile against. `cache_utils.h` and the
    # custom-kernels header only exist for the real GPU runtime.
    set(IMPL_DEPS "")
    if(NOT BUILD_MOCK_RUNTIME)
        list(APPEND COMPILE_FLAGS "-I" "${CMAKE_CURRENT_SOURCE_DIR}/real")

        # Add custom_kernels include directory
        list(APPEND COMPILE_FLAGS "-I" "${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/include")

        # Add ROCm includes
        if(THEROCK_DIST)
            message(STATUS "Adding TheRock includes for ${OUTPUT_BC}: ${THEROCK_DIST}/include")
            list(APPEND COMPILE_FLAGS "-I" "${THEROCK_DIST}/include")
            list(APPEND COMPILE_FLAGS "-D__HIP_PLATFORM_AMD__")
        endif()

        list(APPEND IMPL_DEPS
            ${CMAKE_CURRENT_SOURCE_DIR}/real/runtime_types.h
            ${CMAKE_CURRENT_SOURCE_DIR}/real/cache_utils.h
            ${CMAKE_CURRENT_SOURCE_DIR}/real/zp_unpack_cache.h
            ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/include/hip_custom_kernels.h)
    else()
        list(APPEND COMPILE_FLAGS "-I" "${CMAKE_CURRENT_SOURCE_DIR}/mock")
        list(APPEND IMPL_DEPS
            ${CMAKE_CURRENT_SOURCE_DIR}/mock/runtime_types.h)
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BC}
        COMMAND ${CLANG_EXECUTABLE}
                -c -emit-llvm -std=c++17
                -D_CRT_SECURE_NO_WARNINGS
                # Trim MSVC-CRT symbols the JIT can't resolve: _Init_thread_*
                # (magic statics), sized operator delete, type_info vtable.
                # LlvmIrJit shims the unsized delete; the rest go unused.
                -fno-threadsafe-statics
                -fno-sized-deallocation
                -fno-rtti
                $<$<CONFIG:Debug>:-D_DEBUG>
                $<$<CONFIG:Debug>:-D_ITERATOR_DEBUG_LEVEL=2>
                $<$<CONFIG:Debug>:-O0>
                $<$<NOT:$<CONFIG:Debug>>:-DNDEBUG>
                $<$<NOT:$<CONFIG:Debug>>:-D_ITERATOR_DEBUG_LEVEL=0>
                $<$<NOT:$<CONFIG:Debug>>:-O2>
                ${COMPILE_FLAGS}
                ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
                -o ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BC}
        DEPENDS ${SOURCE_FILE}
                ${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_ep_runtime.h
                ${CMAKE_CURRENT_SOURCE_DIR}/runtime_state_internal.h
                ${CMAKE_CURRENT_SOURCE_DIR}/op_state.h
                ${CMAKE_CURRENT_SOURCE_DIR}/op_profile.h
                ${CMAKE_CURRENT_SOURCE_DIR}/chrome_trace.h
                ${CMAKE_CURRENT_SOURCE_DIR}/debug_log.h
                ${IMPL_DEPS}
                ${CMAKE_BINARY_DIR}/schemas/model_metadata_generated.h
        COMMENT "Compiling ${SOURCE_FILE} to bitcode"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
endmacro()

# Compile shared implementation files
compile_to_bitcode(hipdnn_ep_runtime_state.cpp runtime_state.bc)
compile_to_bitcode(hipdnn_ep_runtime_tensor.cpp runtime_tensor.bc)
compile_to_bitcode(hipdnn_ep_runtime_loop.cpp runtime_loop.bc)
compile_to_bitcode(hipdnn_ep_runtime_if.cpp runtime_if.bc)
compile_to_bitcode(op_profile.cpp runtime_op_profile.bc)
compile_to_bitcode(chrome_trace.cpp runtime_chrome_trace.bc)
compile_to_bitcode(output_allocator.cpp runtime_output_allocator.bc)
# Op-state slots ABI + reference constructors. Shared (host-only allocation),
# so the same TU serves both the real and mock runtimes.
compile_to_bitcode(op_state.cpp runtime_op_state.bc)

# Compile implementation-specific files
if(NOT BUILD_MOCK_RUNTIME)
    # Real GPU runtime implementation
    compile_to_bitcode(real/hip.cpp runtime_hip.bc)
    compile_to_bitcode(real/miopen.cpp runtime_miopen.bc)
    compile_to_bitcode(real/hipblas.cpp runtime_hipblas.bc)
    # Vendor-symbol stubs so a model links without libMIOpen / libhipblaslt (see
    # real/vendor_blas_stub.cpp).
    if(HIPDNN_EP_DISABLE_VENDOR_BLAS)
        compile_to_bitcode(real/vendor_blas_stub.cpp runtime_vendor_blas_stub.bc)
    endif()
	compile_to_bitcode(real/im2col.cpp runtime_im2col.bc)
    compile_to_bitcode(real/memory.cpp runtime_memory.bc)
    compile_to_bitcode(real/elementwise.cpp runtime_elementwise.bc)
    compile_to_bitcode(real/power.cpp runtime_power.bc)
    compile_to_bitcode(real/matmul.cpp runtime_matmul.bc)
    compile_to_bitcode(real/activation.cpp runtime_activation.bc)
    compile_to_bitcode(real/bias_gelu.cpp runtime_bias_gelu.bc)
    compile_to_bitcode(real/fast_gelu.cpp runtime_fast_gelu.bc)
    compile_to_bitcode(real/layer_normalization.cpp runtime_layer_normalization.bc)
    compile_to_bitcode(real/simplified_layer_norm.cpp runtime_simplified_layer_norm.bc)
    compile_to_bitcode(real/skip_simplified_layer_norm.cpp runtime_skip_simplified_layer_norm.bc)
    compile_to_bitcode(real/rotary_embedding.cpp runtime_rotary_embedding.bc)
    compile_to_bitcode(real/gqa.cpp runtime_gqa.bc)
    compile_to_bitcode(real/cast.cpp runtime_cast.bc)
    compile_to_bitcode(real/range.cpp runtime_range.bc)
    compile_to_bitcode(real/gather.cpp runtime_gather.bc)
    compile_to_bitcode(real/gather_elements.cpp runtime_gather_elements.bc)
    compile_to_bitcode(real/top_k.cpp runtime_top_k.bc)
    compile_to_bitcode(real/scatter_elements.cpp runtime_scatter_elements.bc)
    compile_to_bitcode(real/compress.cpp runtime_compress.bc)
    compile_to_bitcode(real/one_hot.cpp runtime_one_hot.bc)
    compile_to_bitcode(real/reduce_sum.cpp runtime_reduce_sum.bc)
    compile_to_bitcode(real/reduce_mean.cpp runtime_reduce_mean.bc)
    compile_to_bitcode(real/reduce_l2.cpp runtime_reduce_l2.bc)
    compile_to_bitcode(real/reduce_max.cpp runtime_reduce_max.bc)
    compile_to_bitcode(real/reduce_min.cpp runtime_reduce_min.bc)
    compile_to_bitcode(real/matmul_nbits.cpp runtime_matmul_nbits.bc)
    compile_to_bitcode(real/qmoe.cpp runtime_qmoe.bc)
    compile_to_bitcode(real/gather_block_quantized.cpp runtime_gather_block_quantized.bc)
    compile_to_bitcode(real/causal_conv_with_state.cpp runtime_causal_conv_with_state.bc)
    compile_to_bitcode(real/test_hip_from_dll.cpp test_hip_from_dll.bc)
    compile_to_bitcode(real/gemm.cpp runtime_gemm.bc)
    compile_to_bitcode(real/where.cpp runtime_where.bc)
    compile_to_bitcode(real/equal.cpp runtime_equal.bc)
    compile_to_bitcode(real/or.cpp runtime_or.bc)
    compile_to_bitcode(real/and.cpp runtime_and.bc)
    compile_to_bitcode(real/div.cpp runtime_div.bc)
    compile_to_bitcode(real/linear_attention.cpp runtime_linear_attention.bc)
    compile_to_bitcode(real/transpose.cpp runtime_transpose.bc)
    compile_to_bitcode(real/abs.cpp runtime_abs.bc)
    compile_to_bitcode(real/neg.cpp runtime_neg.bc)
    compile_to_bitcode(real/not.cpp runtime_not.bc)
    compile_to_bitcode(real/cos.cpp runtime_cos.bc)
    compile_to_bitcode(real/sin.cpp runtime_sin.bc)
    compile_to_bitcode(real/ceil.cpp runtime_ceil.bc)
    compile_to_bitcode(real/exp.cpp runtime_exp.bc)
    compile_to_bitcode(real/log.cpp runtime_log.bc)
    compile_to_bitcode(real/cumsum.cpp runtime_cumsum.bc)
    compile_to_bitcode(real/pad.cpp runtime_pad.bc)
    compile_to_bitcode(real/tile.cpp runtime_tile.bc)
    compile_to_bitcode(real/expand.cpp runtime_expand.bc)
    compile_to_bitcode(real/reduce_prod.cpp runtime_reduce_prod.bc)
    compile_to_bitcode(real/less.cpp runtime_less.bc)
    compile_to_bitcode(real/gather_nd.cpp runtime_gather_nd.bc)
    compile_to_bitcode(real/sign.cpp runtime_sign.bc)
    compile_to_bitcode(real/mod.cpp runtime_mod.bc)
    compile_to_bitcode(real/slice.cpp runtime_slice.bc)
    compile_to_bitcode(real/scatter_nd.cpp runtime_scatter_nd.bc)
    compile_to_bitcode(real/multi_head_attention.cpp runtime_multi_head_attention.bc)
    compile_to_bitcode(real/nonzero.cpp runtime_nonzero.bc)
    compile_to_bitcode(real/size.cpp runtime_size.bc)
    compile_to_bitcode(real/pool.cpp runtime_pool.bc)
    compile_to_bitcode(real/resize.cpp runtime_resize.bc)
    compile_to_bitcode(real/global_pool.cpp runtime_global_pool.bc)

    set(RUNTIME_BC_MODULES
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_state.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_tensor.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_loop.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_if.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_op_profile.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_chrome_trace.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_output_allocator.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_op_state.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_memory.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_hip.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_miopen.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_hipblas.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_im2col.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_elementwise.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_power.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_matmul.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_activation.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_bias_gelu.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_fast_gelu.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_layer_normalization.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_simplified_layer_norm.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_skip_simplified_layer_norm.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_rotary_embedding.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_gqa.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_cast.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_range.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_gather.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_gather_elements.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_top_k.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_scatter_elements.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_compress.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_one_hot.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_reduce_sum.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_reduce_mean.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_reduce_l2.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_reduce_max.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_reduce_min.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_matmul_nbits.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_qmoe.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_causal_conv_with_state.bc
        ${CMAKE_CURRENT_BINARY_DIR}/test_hip_from_dll.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_gemm.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_where.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_equal.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_or.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_and.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_div.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_linear_attention.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_transpose.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_abs.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_neg.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_not.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_cos.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_sin.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_ceil.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_exp.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_log.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_cumsum.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_pad.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_tile.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_expand.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_reduce_prod.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_less.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_gather_nd.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_sign.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_mod.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_slice.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_scatter_nd.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_multi_head_attention.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_nonzero.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_size.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_pool.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_resize.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_global_pool.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_gather_block_quantized.bc
    )
    # Vendor-symbol stubs (see the compile_to_bitcode above) join runtime.bc only
    # when the vendor BLAS/DNN backends are disabled.
    if(HIPDNN_EP_DISABLE_VENDOR_BLAS)
        list(APPEND RUNTIME_BC_MODULES
            ${CMAKE_CURRENT_BINARY_DIR}/runtime_vendor_blas_stub.bc)
    endif()
else()
    # Mock runtime for testing without GPU
    compile_to_bitcode(mock/mock_gpu.cpp runtime_mock.bc)
    compile_to_bitcode(mock/memory.cpp runtime_memory.bc)
    compile_to_bitcode(mock/test_hip_from_dll.cpp test_hip_from_dll.bc)
    # Mock per-op state constructors (real ones live in each op's real/ TU).
    compile_to_bitcode(mock/op_state_stubs.cpp runtime_op_state_stubs.bc)

    set(RUNTIME_BC_MODULES
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_state.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_tensor.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_loop.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_if.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_op_profile.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_chrome_trace.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_output_allocator.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_op_state.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_op_state_stubs.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_memory.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_mock.bc
        ${CMAKE_CURRENT_BINARY_DIR}/test_hip_from_dll.bc
    )
endif()

# ============================================================================
# STEP 4: Link Bitcode Modules into Single runtime.bc
# ============================================================================

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc

    # Combine all bitcode modules into single runtime.bc
    COMMAND ${LLVM_LINK_EXECUTABLE}
            ${RUNTIME_BC_MODULES}
            -o ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc

    DEPENDS ${RUNTIME_BC_MODULES}

    COMMENT "Linking runtime modules into runtime.bc"
    VERBATIM
)

# ============================================================================
# STEP 5: Create Custom Target for Runtime Bitcode
# ============================================================================

add_custom_target(HipRuntime
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc
)

# Export runtime.bc path for consumers
set(RUNTIME_BC_PATH ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc PARENT_SCOPE)

add_custom_target(RuntimeBitcode DEPENDS HipRuntime)

# ============================================================================
# STEP 6: Native current-stream accessor (NOT bitcode)
# ============================================================================
# The per-thread session-stream slot is compiled natively here, deliberately
# kept out of runtime.bc. A thread_local inside runtime.bc would force the JIT
# to lower emulated TLS (__emutls_get_address), which has no definition in the
# JIT process. Instead the JIT'd runtime calls these accessors as ordinary
# external functions: the host (EP DLL / hip-test / hip-inspect) defines them
# (see LlvmIrJit absolute-symbol registration), and native model DLLs link
# their own copy of this lib (CompilerDriver::discoverLibraries).
add_library(hipdnn-ep-tls STATIC tls_stream.cpp)
set_target_properties(hipdnn-ep-tls PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hipdnn-ep-tls PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Install the import/static lib so NATIVE model DLLs can link it on a deploy
# host that only has the packaged tree (no build dir). CompilerDriver bakes the
# build-tree path into HIPDNN_EP_TLS_LIB_DIR, which does not exist on the GPU
# test machine; staging hipdnn-ep-tls.lib into install/lib (then onto LIB) lets
# lld-link resolve it there. Bitcode (default) does not need it.
install(TARGETS hipdnn-ep-tls ARCHIVE DESTINATION lib)

# ============================================================================
# STEP 7: Shared embedded runtime.bc (single source of truth)
# ============================================================================
# Both consumers need runtime.bc embedded as bytes:
#   - the compiler (LLVMBackend::linkRuntimeModule, native producer path)
#   - the EP / tools (LlvmIrJit, JIT load path)
# Generate the C array ONCE here and expose it as a static lib both link,
# instead of each target running its own xxd step. This keeps a single source
# of truth and removes the latent duplicate-symbol hazard if a binary ever
# links both consumers (each previously defined its own runtime_bc_data).
# Note: each consuming binary still embeds its own copy (separate DLLs); this
# consolidates the build logic, not the per-binary bytes.
if(TARGET RuntimeBitcode)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/cmake/xxd.py
                --var runtime_bc_data
                --output ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp
                ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc
        COMMAND ${Python3_EXECUTABLE} -c
                "import os; data = open('${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp').read(); data = data.replace('static const unsigned char', 'extern \"C\" const unsigned char'); bc_size = os.path.getsize('${CMAKE_CURRENT_BINARY_DIR}/runtime.bc'); data = '#include <cstddef>\\n' + data + f'\\nextern \"C\" const size_t runtime_bc_data_size = {bc_size};\\n'; open('${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp', 'w').write(data)"
        COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp.tmp
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc RuntimeBitcode
        COMMENT "Embedding runtime.bc (shared) via runtime_ir_data.cpp"
        VERBATIM
    )
else()
    # No clang at configure time -> empty bytes; consumers treat size 0 as a
    # degraded/fatal mode but unrelated targets still build.
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp
        "#include <cstddef>\n"
        "extern \"C\" const unsigned char runtime_bc_data[] = {0};\n"
        "extern \"C\" const size_t runtime_bc_data_size = 0;\n"
    )
endif()

add_library(runtime-bc-embed STATIC ${CMAKE_CURRENT_BINARY_DIR}/runtime_ir_data.cpp)
set_target_properties(runtime-bc-embed PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET RuntimeBitcode)
    add_dependencies(runtime-bc-embed RuntimeBitcode)
endif()

message(STATUS "Runtime library configuration:")
message(STATUS "  Mode: ${BUILD_MOCK_RUNTIME}")
message(STATUS "  Runtime bitcode will be at: ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc")
