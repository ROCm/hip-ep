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

# ==============================================================================
# Runtime Configuration
# ==============================================================================
option(BUILD_MOCK_RUNTIME "Build mock runtime (no GPU required)" ON)

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
# STEP 2: Generate LLVM Bitcode from Runtime Modules
# ============================================================================

# Macro to compile a single C++ file to bitcode
macro(compile_to_bitcode SOURCE_FILE OUTPUT_BC)
    # Build include flags
    set(COMPILE_FLAGS "")

    # Always add root directory for public headers
    list(APPEND COMPILE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

    # Add morphizen-foundation headers
    list(APPEND COMPILE_FLAGS "-I${MORPHIZEN_FOUNDATION_INCLUDE_DIR}")

    # Add schemas generated headers (model_metadata_generated.h etc.)
    list(APPEND COMPILE_FLAGS "-I${CMAKE_BINARY_DIR}/schemas")

    # Add flatbuffers headers
    get_target_property(_fb_incs flatbuffers::flatbuffers INTERFACE_INCLUDE_DIRECTORIES)
    if(_fb_incs)
        foreach(_inc ${_fb_incs})
            list(APPEND COMPILE_FLAGS "-I${_inc}")
        endforeach()
    endif()

    # Add Mock implementation directory
    list(APPEND COMPILE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/mock")

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

# ============================================================================
# STEP 3: Compile Runtime Modules Based on Mode
# ============================================================================

if(BUILD_MOCK_RUNTIME)
    # Compile shared implementation files
    compile_to_bitcode(hipdnn_ep_runtime_state.cpp runtime_state.bc)
    compile_to_bitcode(hipdnn_ep_runtime_tensor.cpp runtime_tensor.bc)

    # Compile Mock runtime backend
    compile_to_bitcode(mock/mock_gpu.cpp runtime_mock.bc)
    compile_to_bitcode(mock/memory.cpp runtime_memory.bc)
    compile_to_bitcode(mock/test_hip_from_dll.cpp test_hip_from_dll.bc)

    # List of all bitcode modules to link
    set(RUNTIME_BC_MODULES
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_state.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_tensor.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_memory.bc
        ${CMAKE_CURRENT_BINARY_DIR}/runtime_mock.bc
        ${CMAKE_CURRENT_BINARY_DIR}/test_hip_from_dll.bc
    )

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

    message(STATUS "Mock Runtime configuration:")
    message(STATUS "  Runtime bitcode will be at: ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc")
    message(STATUS "  Output size: ~84KB (5 modules)")
endif()

if(NOT BUILD_MOCK_RUNTIME)
    # Real Runtime implementation will be added in future PRs
    # PR 2: Custom HIP Kernels Library
    # PR 3: Real Runtime Implementation
endif()

# ============================================================================
# KEEP OLD hip_runtime_static for backward compatibility
# ============================================================================

file(GLOB RUNTIME_SRCS "ops_runtime/*.cpp")
add_library(hip_runtime_static STATIC ${RUNTIME_SRCS})
set_target_properties(hip_runtime_static PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)
target_compile_definitions(hip_runtime_static PRIVATE __HIP_PLATFORM_AMD__ MIOPEN_BETA_API)
if(DEFINED ENV{THEROCK_DIST})
  target_include_directories(hip_runtime_static PRIVATE $ENV{THEROCK_DIST}/include)
endif()
