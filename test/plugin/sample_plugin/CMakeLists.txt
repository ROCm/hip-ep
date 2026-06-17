##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Sample plugin DLL exercising the public plugin ABI surface in CI.
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

add_library(hip_ep_sample_plugin SHARED
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

# Plugin links against the MLIR libraries it actually uses
# (PassRegistration<>, PassWrapper<>, func::FuncOp). It does NOT link
# LibHipCompiler -- a real vendor plugin is a downstream binary
# consumer of the public hip-compiler artefact, not an internal
# component, so we match that constraint here in the in-tree sample.
#
# Bringing in MLIRPass also brings the MLIRSupport / MLIRIR transitive
# dependencies the pass needs.
target_link_libraries(hip_ep_sample_plugin PRIVATE
  MLIRPass
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
)

# The plugin uses public hip-compiler headers (PluginAPI.h,
# PluginRegistry.h). Both already on the global include path via
# the top-level include_directories(); no per-target setup needed.

# Pull in the LLVM_ATTRIBUTE_WEAK / Compiler.h shim from LLVM.
target_include_directories(hip_ep_sample_plugin PRIVATE
  ${LLVM_INCLUDE_DIRS}
)

# Windows: export hipEpGetPluginInfo without a .def file.
set_target_properties(hip_ep_sample_plugin PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  FOLDER "mlir/Test/Plugin"
  # Keep the file name predictable for the test runner; cmake's
  # default would prepend "lib" on non-Windows.
  PREFIX ""
  OUTPUT_NAME "hip_ep_sample_plugin"
)

# C++17 to match the rest of the tree.
target_compile_features(hip_ep_sample_plugin PRIVATE cxx_std_17)
