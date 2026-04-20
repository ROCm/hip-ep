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

# Clang REQUIRED for bitcode generation
find_program(CLANG_EXECUTABLE NAMES clang clang-cl
    PATHS
        ${LLVM_BINARY_DIR}/bin       # When LLVM built from source
        ${LLVM_TOOLS_BINARY_DIR}     # When LLVM installed via package
)

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

# llvm-link REQUIRED for combining bitcode files
find_program(LLVM_LINK_EXECUTABLE NAMES llvm-link
    PATHS
        ${LLVM_BINARY_DIR}/bin
        ${LLVM_TOOLS_BINARY_DIR}
)

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

# Default to Mock runtime (no GPU required)
option(BUILD_MOCK_RUNTIME "Build mock runtime (no GPU required)" ON)

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
    find_package(MIOpen QUIET)
    find_package(hipblaslt QUIET)

    if(DEFINED THEROCK_DIST)
        set(CMAKE_PREFIX_PATH "${ORIGINAL_CMAKE_PREFIX_PATH}")
    endif()

    if(NOT hip_FOUND OR NOT MIOpen_FOUND OR NOT hipblaslt_FOUND)
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

    # Add Memory Manager headers
    list(APPEND COMPILE_FLAGS "-I" "${CMAKE_SOURCE_DIR}/lib/MemoryManager")

    # Add implementation directory
    if(NOT BUILD_MOCK_RUNTIME)
        list(APPEND COMPILE_FLAGS "-I" "${CMAKE_CURRENT_SOURCE_DIR}/real")

        # Add custom_kernels include directory
        list(APPEND COMPILE_FLAGS "-I" "${CMAKE_SOURCE_DIR}/3rd-party/custom_kernels/include")

        # Add ROCm includes
        if(THEROCK_DIST)
            message(STATUS "Adding TheRock includes for ${OUTPUT_BC}: ${THEROCK_DIST}/include")
            list(APPEND COMPILE_FLAGS "-I" "${THEROCK_DIST}/include")
            list(APPEND COMPILE_FLAGS "-D__HIP_PLATFORM_AMD__")
            list(APPEND COMPILE_FLAGS "-DMM_HAS_HIP")
        endif()
    else()
        list(APPEND COMPILE_FLAGS "-I" "${CMAKE_CURRENT_SOURCE_DIR}/mock")
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_BC}
        COMMAND ${CLANG_EXECUTABLE}
                -c -emit-llvm -std=c++17
                -D_CRT_SECURE_NO_WARNINGS
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
                ${CMAKE_BINARY_DIR}/schemas/model_metadata_generated.h
        COMMENT "Compiling ${SOURCE_FILE} to bitcode"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
endmacro()

# Compile shared implementation files
compile_to_bitcode(hipdnn_ep_runtime_state.cpp runtime_state.bc)
compile_to_bitcode(hipdnn_ep_runtime_tensor.cpp runtime_tensor.bc)

# Compile implementation-specific files
if(NOT BUILD_MOCK_RUNTIME)
    # Real GPU runtime implementation
    compile_to_bitcode(real/hip.cpp runtime_hip.bc)
    compile_to_bitcode(real/miopen.cpp runtime_miopen.bc)
    compile_to_bitcode(real/hipblas.cpp runtime_hipblas.bc)
    compile_to_bitcode(real/memory.cpp runtime_memory.bc)
    compile_to_bitcode(real/elementwise.cpp runtime_elementwise.bc)
    compile_to_bitcode(real/matmul.cpp runtime_matmul.bc)
    compile_to_bitcode(real/activation.cpp runtime_activation.bc)
    compile_to_bitcode(real/simplified_layer_norm.cpp runtime_simplified_layer_norm.bc)
    compile_to_bitcode(real/skip_simplified_layer_norm.cpp runtime_skip_simplified_layer_norm.bc)
    compile_to_bitcode(real/rotary_embedding.cpp runtime_rotary_embedding.bc)
    compile_to_bitcode(real/gqa.cpp runtime_gqa.bc)
    compile_to_bitcode(real/cast.cpp runtime_cast.bc)
    compile_to_bitcode(real/gather.cpp runtime_gather.bc)
    compile_to_bitcode(real/reduce_sum.cpp runtime_reduce_sum.bc)
    compile_to_bitcode(real/matmul_nbits.cpp runtime_matmul_nbits.bc)
    compile_to_bitcode(real/qmoe.cpp runtime_qmoe.bc)
    compile_to_bitcode(real/test_hip_from_dll.cpp test_hip_from_dll.bc)
    compile_to_bitcode(real/gemm.cpp runtime_gemm.bc)

    # Memory Manager modules (compiled to bitcode for DLL linking)
    compile_to_bitcode(../MemoryManager/mm_hal.cpp mm_hal.bc)
    compile_to_bitcode(../MemoryManager/mm_hal_host.cpp mm_hal_host.bc)
    compile_to_bitcode(../MemoryManager/mm_hal_hip.cpp mm_hal_hip.bc)
    compile_to_bitcode(../MemoryManager/mm_handle_table.cpp mm_handle_table.bc)
    compile_to_bitcode(../MemoryManager/mm_config.cpp mm_config.bc)
    compile_to_bitcode(../MemoryManager/mm_core.cpp mm_core.bc)
    compile_to_bitcode(../MemoryManager/mm_pool.cpp mm_pool.bc)
    compile_to_bitcode(../MemoryManager/mm_arena.cpp mm_arena.bc)
    compile_to_bitcode(../MemoryManager/mm_runtime_bridge.cpp mm_runtime_bridge.bc)
    compile_to_bitcode(../MemoryManager/mm_free_list.cpp mm_free_list.bc)
    compile_to_bitcode(../MemoryManager/mm_kv.cpp mm_kv.bc)
    compile_to_bitcode(../MemoryManager/mm_block_table.cpp mm_block_table.bc)

    set(RUNTIME_BC_MODULES
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_state.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_tensor.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_memory.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_hip.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_miopen.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_hipblas.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_elementwise.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_matmul.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_activation.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_simplified_layer_norm.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_skip_simplified_layer_norm.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_rotary_embedding.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_gqa.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_cast.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_gather.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_reduce_sum.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_matmul_nbits.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_qmoe.bc
        ${CMAKE_CURRENT_BINARY_DIR}/test_hip_from_dll.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_gemm.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_hal.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_hal_host.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_hal_hip.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_handle_table.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_config.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_core.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_pool.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_arena.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_runtime_bridge.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_free_list.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_kv.bc
        ${CMAKE_CURRENT_BINARY_DIR}/mm_block_table.bc
    )
else()
    # Mock runtime for testing without GPU
    compile_to_bitcode(mock/mock_gpu.cpp runtime_mock.bc)
    compile_to_bitcode(mock/memory.cpp runtime_memory.bc)
    compile_to_bitcode(mock/test_hip_from_dll.cpp test_hip_from_dll.bc)

    set(RUNTIME_BC_MODULES
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_state.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_tensor.bc
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

message(STATUS "Runtime library configuration:")
message(STATUS "  Mode: ${BUILD_MOCK_RUNTIME}")
message(STATUS "  Runtime bitcode will be at: ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc")
