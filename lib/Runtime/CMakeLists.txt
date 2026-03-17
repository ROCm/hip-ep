##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# HipDnnEpRuntime - Runtime library for ONNX HIP DNN Execution Provider
# Provides HIP/MIOpen/hipBLASLt wrappers called by generated LLVM IR
#
# NOTE: "HipDnnEP" = HIP DNN Execution Provider (this project)
#       "hipDNN" = AMD's separate DNN library (different project - not used here)
#
# CRITICAL BUILD FLOW:
# This CMakeLists.txt implements a non-standard two-stage compilation:
#
# Stage 1 (Build Time - Once):
#   hipdnn_ep_runtime.cpp → [Clang -emit-llvm] → runtime.bc → [xxd.py] → runtime_ir_data.cpp
#   Result: Runtime compiled to LLVM bitcode and embedded as C array in libHipDnnEpRuntime.a
#
# Stage 2 (Model Compilation - Per Model):
#   Generated MLIR IR + embedded runtime.bc → [llvm::Linker] → Combined IR → [LLVM O2] → model.dll
#   Result: Runtime functions inlined into generated code (zero-cost abstraction)
#
# Why this is necessary:
# - Enables cross-module inlining (generated code + Runtime implementation)
# - Accessor functions (hipdnn_ep_get_stream, etc.) become single load instructions
# - Zero runtime overhead while maintaining clean abstraction
# - See: ARCHITECTURE.md Design Decision #7 (LLVM IR Merging)

cmake_minimum_required(VERSION 3.18)

# ============================================================================
# STEP 1: Find Required Tools
# ============================================================================

# Python 3 required for xxd.py (embeds binary data as C array)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Clang REQUIRED for bitcode generation
# Why Clang specifically?
#   - MSVC (cl.exe) doesn't support -emit-llvm flag
#   - GCC's -emit-llvm produces incompatible format
#   - Clang is the reference LLVM frontend
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

# llvm-link REQUIRED for combining multiple bitcode files
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
# STEP 1b: Find morphizen-foundation headers (needed by state.cpp for
# hipdnn_ep_state_init_with_fs which uses morphizen::FileSystem*)
# The header is vendored in-tree under include/morphizen-foundation/; fall back
# to CMAKE_PREFIX_PATH for external installations.
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
# STEP 2: Find ROCm Components (Optional - Mock Mode Available)
# ============================================================================

# ROCm components are optional for development/testing
# If not found, build mock runtime (prints debug messages, no GPU calls)
# Controlled by BUILD_MOCK_RUNTIME variable (default: auto-detect)

if(NOT BUILD_MOCK_RUNTIME)
    # If THEROCK_DIST is set, add it to CMAKE_PREFIX_PATH temporarily
    # This avoids polluting the main project's LLVM search
    if(DEFINED ENV{THEROCK_DIST})
        set(THEROCK_DIST "$ENV{THEROCK_DIST}")
        message(STATUS "Using TheRock ROCm SDK at: ${THEROCK_DIST}")

        # Save original CMAKE_PREFIX_PATH
        set(ORIGINAL_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")

        # Temporarily add TheRock to search path for ROCm packages
        list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")
        list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")
    endif()

    find_package(hip QUIET)         # HIP runtime (hipMalloc, hipMemcpy, etc.)
    find_package(MIOpen QUIET)      # DNN operations (convolution)
    find_package(hipblaslt QUIET)   # BLAS operations (matrix multiply)

    # Restore original CMAKE_PREFIX_PATH to avoid affecting other targets
    if(DEFINED THEROCK_DIST)
        set(CMAKE_PREFIX_PATH "${ORIGINAL_CMAKE_PREFIX_PATH}")
    endif()

    if(NOT hip_FOUND OR NOT MIOpen_FOUND OR NOT hipblaslt_FOUND)
        message(WARNING "ROCm components not found. Building mock runtime library for testing.")
        message(WARNING "  hip_FOUND=${hip_FOUND}, MIOpen_FOUND=${MIOpen_FOUND}, hipblaslt_FOUND=${hipblaslt_FOUND}")
        message(WARNING "  Set THEROCK_DIST environment variable to ROCm SDK path")
        message(WARNING "  Set HIP_PLATFORM=amd environment variable")
        message(WARNING "  To build full runtime, install ROCm and ensure it's in CMAKE_PREFIX_PATH")
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

# WHY THIS STEP IS NECESSARY:
# We need Runtime implementation as LLVM bitcode so it can be merged with
# generated code during model compilation. This enables the LLVM optimizer
# to inline Runtime functions across module boundaries.
#
# Example transformation after IR merging + optimization:
#
#   Before (generated code):
#     %stream = call ptr @hipdnn_ep_get_stream(ptr %state)
#
#   After (inlined - zero overhead):
#     %stream_ptr = getelementptr inbounds %struct.RuntimeState, ptr %state, i32 0, i32 0
#     %stream = load ptr, ptr %stream_ptr
#
# This is the "zero-cost abstraction" - clean code with no performance penalty.

# Macro to compile a single C++ file to bitcode
# For shared files (state.cpp, constants.cpp, tensor.cpp): SOURCE_FILE is in root
# For impl-specific files: SOURCE_FILE is relative to IMPL_DIR (mock/ or real/)
macro(compile_to_bitcode SOURCE_FILE OUTPUT_BC)
    # Build include flags
    set(COMPILE_FLAGS "")

    # Always add root directory for public headers
    list(APPEND COMPILE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

    # Add morphizen-foundation headers (for FileSystem* in state.cpp)
    list(APPEND COMPILE_FLAGS "-I${MORPHIZEN_FOUNDATION_INCLUDE_DIR}")

    # Add proto generated headers (model_metadata_generated.h etc.)
    list(APPEND COMPILE_FLAGS "-I${CMAKE_BINARY_DIR}/proto")

    # Add flatbuffers headers (for flatbuffers/flatbuffers.h used by generated headers)
    get_target_property(_fb_incs flatbuffers::flatbuffers INTERFACE_INCLUDE_DIRECTORIES)
    if(_fb_incs)
        foreach(_inc ${_fb_incs})
            list(APPEND COMPILE_FLAGS "-I${_inc}")
        endforeach()
    endif()

    # Add implementation directory to include path for runtime_types.h
    # This is what enables the abstraction - compiler finds the right runtime_types.h
    if(NOT BUILD_MOCK_RUNTIME)
        list(APPEND COMPILE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/real")

        # Add ROCm includes for real runtime
        if(THEROCK_DIST)
            message(STATUS "Adding TheRock includes for ${OUTPUT_BC}: ${THEROCK_DIST}/include")
            list(APPEND COMPILE_FLAGS "-I${THEROCK_DIST}/include")
            list(APPEND COMPILE_FLAGS "-D__HIP_PLATFORM_AMD__")
        else()
            message(WARNING "THEROCK_DIST not set in macro for ${OUTPUT_BC}")
        endif()

        # Add custom kernels header for RoPE/GQA wrappers
        if(hip_CUSTOM_KERNELS_INCLUDE_DIR)
            list(APPEND COMPILE_FLAGS "-I${hip_CUSTOM_KERNELS_INCLUDE_DIR}")
        endif()
    else()
        list(APPEND COMPILE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/mock")
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
        # model_metadata_generated.h is produced by the UdnaSchemas flatc step;
        # list it explicitly so Ninja waits for flatc before running clang.
        DEPENDS ${SOURCE_FILE} hipdnn_ep_runtime.h
                ${CMAKE_BINARY_DIR}/proto/model_metadata_generated.h
        COMMENT "Compiling ${SOURCE_FILE} to bitcode"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
endmacro()

# Compile shared implementation files (same for mock and real)
# These include "runtime_types.h" which resolves to mock/ or real/ based on include path
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
    compile_to_bitcode(real/test_hip_from_dll.cpp test_hip_from_dll.bc)

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
        ${CMAKE_CURRENT_BINARY_DIR}/test_hip_from_dll.bc
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

# Runtime library's responsibility ends at runtime.bc generation.
# The Backend project will consume runtime.bc and convert it to C array as needed.

add_custom_target(UdnaRuntime
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc
)

# Export runtime.bc path for consumers (Backend project)
set(RUNTIME_BC_PATH ${CMAKE_CURRENT_BINARY_DIR}/runtime.bc PARENT_SCOPE)

# Alias for backward compatibility
add_custom_target(RuntimeBitcode DEPENDS UdnaRuntime)

# ============================================================================
# SUMMARY OF BUILD ARTIFACTS AND PIPELINE
# ============================================================================
#
# Runtime project output:
#   - runtime.bc                   - LLVM bitcode (final artifact)
#
# How runtime.bc is consumed (by Backend project):
#   1. Backend/CMakeLists.txt runs xxd.py to convert runtime.bc → runtime_ir_data.cpp
#   2. runtime_ir_data.cpp contains: runtime_bc_data[] array + runtime_bc_data_size
#   3. Backend/LLVMBackend.cpp includes runtime_ir_data.cpp as source file
#   4. LLVMBackend.lib (containing runtime_bc_data) is linked into morphizen-compile.exe
#
# What happens at model compilation (morphizen-compile):
#   1. LLVMBackend::linkRuntimeModule() reads runtime_bc_data from its own memory
#   2. Parses bitcode to LLVM IR using llvm::parseBitcodeFile()
#   3. Merges runtime IR with generated model IR using llvm::Linker::linkInModule()
#   4. LLVM optimizer inlines runtime functions into model code (zero-cost abstraction)
#   5. Result: model.dll contains inlined runtime code, no separate runtime library needed
#
# Key insight: Runtime code flows through LLVM IR, not through traditional linking:
#   C++ source → bitcode → IR merge → optimization → model.dll
#
# See: docs/RUNTIME-ARCHITECTURE.md for detailed architecture explanation
# ============================================================================
