##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Sample static plugin exercising the public plugin surface in CI.
# See docs/design/plugin-interface.md for the broader context.

# ============================================================================
# Embed a tiny LLVM bitcode buffer in the plugin DLL.
#
# The plugin's RegisterCallbacks hands this buffer to hip-compiler via
# HipEpPluginRegistry::addRuntimeBitcode; LLVMBackend::linkRuntimeModule
# then links it into the model module with Linker::Flags::OverrideFromSrc.
# The mechanism mirrors lib/Runtime/CMakeLists.txt's "compile-to-bitcode +
# xxd embed" pipeline, just at a fraction of the scale.
#
# If clang is not on PATH at configure time we fall back to embedding an
# empty buffer; the plugin still compiles and the unit test detects the
# degraded mode. This matches lib/Target/LLVM/CMakeLists.txt's behaviour
# for the in-tree runtime.
# ============================================================================

find_program(SAMPLE_PLUGIN_CLANG NAMES clang clang-cl
    PATHS ${LLVM_BINARY_DIR}/bin ${LLVM_TOOLS_BINARY_DIR}
)
find_package(Python3 QUIET COMPONENTS Interpreter)
if(NOT Python3_FOUND)
    find_program(Python3_EXECUTABLE python)
endif()

set(SAMPLE_PLUGIN_BITCODE_SRC ${CMAKE_CURRENT_BINARY_DIR}/sample_plugin_bitcode.cpp)

if(SAMPLE_PLUGIN_CLANG AND Python3_EXECUTABLE)
    set(SAMPLE_PLUGIN_BC ${CMAKE_CURRENT_BINARY_DIR}/sample_plugin_runtime.bc)

    add_custom_command(
        OUTPUT ${SAMPLE_PLUGIN_BC}
        COMMAND ${SAMPLE_PLUGIN_CLANG}
                -c -emit-llvm -std=c++17 -O2
                ${CMAKE_CURRENT_SOURCE_DIR}/sample_plugin_runtime.cpp
                -o ${SAMPLE_PLUGIN_BC}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sample_plugin_runtime.cpp
        COMMENT "Compiling sample_plugin_runtime.cpp to bitcode"
        VERBATIM
    )

    # xxd.py emits a `static const unsigned char <var>[]` array; we
    # patch it to `extern "C"` linkage so sample_plugin.cpp (a separate
    # TU) can pick it up via an `extern "C" const unsigned char ...[]`
    # declaration. Mirrors lib/Target/LLVM/CMakeLists.txt's recipe for
    # runtime_bc_data.
    add_custom_command(
        OUTPUT ${SAMPLE_PLUGIN_BITCODE_SRC}
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/cmake/xxd.py
                --var kSamplePluginBitcode
                --output ${SAMPLE_PLUGIN_BITCODE_SRC}.tmp
                ${SAMPLE_PLUGIN_BC}
        COMMAND ${Python3_EXECUTABLE} -c
                "import os; data = open('${SAMPLE_PLUGIN_BITCODE_SRC}.tmp').read(); data = data.replace('static const unsigned char', 'extern \"C\" const unsigned char'); bc_size = os.path.getsize('${SAMPLE_PLUGIN_BC}'); data = '#include <cstddef>\\n' + data + f'\\nextern \"C\" const std::size_t kSamplePluginBitcodeSize = {bc_size};\\n'; open('${SAMPLE_PLUGIN_BITCODE_SRC}', 'w').write(data)"
        COMMAND ${CMAKE_COMMAND} -E remove ${SAMPLE_PLUGIN_BITCODE_SRC}.tmp
        DEPENDS ${SAMPLE_PLUGIN_BC}
        COMMENT "Embedding sample_plugin_runtime.bc into sample_plugin_bitcode.cpp"
        VERBATIM
    )
else()
    # Degraded mode: emit an empty buffer so the plugin still compiles.
    # Unit test detects the empty buffer and skips the round-trip
    # assertion with an explanatory message.
    message(STATUS
        "Sample plugin: clang or python not found; "
        "embedding empty bitcode buffer "
        "(SAMPLE_PLUGIN_CLANG=${SAMPLE_PLUGIN_CLANG}, "
        "Python3_EXECUTABLE=${Python3_EXECUTABLE})")
    file(WRITE ${SAMPLE_PLUGIN_BITCODE_SRC}
        "// Auto-generated stub: clang/python not available at configure time.\n"
        "#include <cstddef>\n"
        "extern \"C\" const unsigned char kSamplePluginBitcode[1] = {0};\n"
        "extern \"C\" const std::size_t kSamplePluginBitcodeSize = 0;\n"
    )
endif()

# ============================================================================
# Build a tiny static library and have the plugin contribute
# its path + name through addLibraryPath / addLibrary. The library
# exposes a single uniquely-named C symbol so it cannot collide with
# anything in-tree.
#
# We deliberately make this a STATIC library (matching how vendor
# kernel packs are typically distributed) so the plugin DLL itself
# does not depend on it at load time.
# ============================================================================
add_library(hip_ep_sample_lib STATIC sample_lib.cpp)
target_compile_features(hip_ep_sample_lib PRIVATE cxx_std_17)
# Platform defaults give us hip_ep_sample_lib.lib (Windows) or
# libhip_ep_sample_lib.a (Linux); both are what lld-link / the
# clang driver search for given the bare name "hip_ep_sample_lib".
set_target_properties(hip_ep_sample_lib PROPERTIES
  FOLDER "mlir/Test/Plugin"
)

add_library(hip_ep_sample_plugin STATIC
  sample_plugin.cpp
  ${SAMPLE_PLUGIN_BITCODE_SRC}
)
add_dependencies(hip_ep_sample_plugin hip_ep_sample_lib)

# Generator expressions resolve at build time, so the directory
# embedded into the plugin source is the canonical
# $<TARGET_FILE_DIR:hip_ep_sample_lib> -- typically the build's
# bin/ directory on this tree's layout.
target_compile_definitions(hip_ep_sample_plugin PRIVATE
  "HIP_EP_SAMPLE_LIB_DIR=\"$<TARGET_FILE_DIR:hip_ep_sample_lib>\""
  "HIP_EP_SAMPLE_LIB_NAME=\"hip_ep_sample_lib\""
)

# MLIR linkage under STATIC plugins: the plugin is a static lib linked into the
# host (when `sample` is in HIPDNN_EP_COMPILER_PLUGINS -> pulled into
# LibHipCompiler by hipdnn_ep_finalize_static_plugins, then into hip-mlir-opt /
# hip-compiler / the EP DLL). Because plugin and host are ONE binary sharing one
# MLIR instance, the registration lands in the host's single MLIR registry with
# no symbol export, identically on Windows and Linux.
#
# The plugin declares the MLIR libraries it references (PUBLIC) so that on a
# static link they follow the plugin archive on the command line -- otherwise
# the plugin's references to mlir:: symbols (PassRegistration, the Pass vtable,
# op accessors) are seen AFTER the MLIR archives already passed and fail to
# resolve. This is a link-ORDER requirement of static archives; it does NOT give
# the plugin a private MLIR instance (the whole host is one binary). It links no
# hip-compiler symbol.
target_include_directories(hip_ep_sample_plugin PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(hip_ep_sample_plugin PUBLIC
  MLIRPass
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
)

set_target_properties(hip_ep_sample_plugin PROPERTIES
  FOLDER "mlir/Test/Plugin"
  # Linked into LibHipCompiler, which is itself linked into the EP DLL (a shared
  # library), so the objects must be position-independent on ELF.
  POSITION_INDEPENDENT_CODE ON
)

# C++17 to match the rest of the tree.
target_compile_features(hip_ep_sample_plugin PRIVATE cxx_std_17)

# Build with RTTI OFF on ELF to match LLVM/MLIR's default no-RTTI ABI (the same
# setting the host tools use). MLIR provides TypeID without RTTI
# (MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID), so the plugin needs none.
# Windows keeps the toolchain default (matched by the host there).
if(NOT WIN32)
  target_compile_options(hip_ep_sample_plugin PRIVATE -fno-rtti)
endif()

# Make this plugin AVAILABLE to the static registrar. It is only LINKED when
# `sample` is listed in HIPDNN_EP_COMPILER_PLUGINS (see cmake/HipEpPlugins.cmake).
hipdnn_ep_compiler_plugin_register(PLUGIN_ID sample TARGET hip_ep_sample_plugin)
