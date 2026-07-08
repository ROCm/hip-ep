##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Sample static plugin exercising the public plugin surface in CI.
# See docs/design/plugin-interface.md for the broader context.

# Embed a tiny LLVM bitcode buffer in the plugin (registration hands it to
# addRuntimeBitcode -> linked into the model module with OverrideFromSrc).
# Mirrors lib/Runtime's "compile-to-bitcode + xxd embed" at small scale. If
# clang/python are absent we embed an empty buffer; the unit test detects it.

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

    # xxd.py emits `static const unsigned char <var>[]`; patch to extern "C"
    # so sample_plugin.cpp (a separate TU) can pick it up.
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
    # Degraded mode: empty buffer so the plugin still compiles (unit test
    # detects it and skips the round-trip).
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

# Tiny STATIC library the plugin contributes via addLibraryPath / addLibrary
# (matches how vendor kernel packs ship). Single uniquely-named symbol.
add_library(hip_ep_sample_lib STATIC sample_lib.cpp)
target_compile_features(hip_ep_sample_lib PRIVATE cxx_std_17)
set_target_properties(hip_ep_sample_lib PROPERTIES
  FOLDER "mlir/Test/Plugin"
)

add_library(hip_ep_sample_plugin STATIC
  sample_plugin.cpp
  ${SAMPLE_PLUGIN_BITCODE_SRC}
)
add_dependencies(hip_ep_sample_plugin hip_ep_sample_lib)

# Embed the sample lib's dir + name (resolved at build time) for the plugin's
# addLibraryPath / addLibrary calls.
target_compile_definitions(hip_ep_sample_plugin PRIVATE
  "HIP_EP_SAMPLE_LIB_DIR=\"$<TARGET_FILE_DIR:hip_ep_sample_lib>\""
  "HIP_EP_SAMPLE_LIB_NAME=\"hip_ep_sample_lib\""
)

# Declare the MLIR libs the plugin references (PUBLIC) so a static link resolves
# them in order -- the plugin's mlir:: refs must follow it on the link line.
# This is link ORDER only; plugin + host are one binary sharing one MLIR (linked
# into LibHipCompiler when `sample` is selected). It links no hip-compiler symbol.
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

# RTTI off on ELF to match LLVM/MLIR's no-RTTI ABI (MLIR TypeID needs none).
if(NOT WIN32)
  target_compile_options(hip_ep_sample_plugin PRIVATE -fno-rtti)
endif()

# Make this plugin AVAILABLE to the static registrar. It is only LINKED when
# `sample` is listed in HIPDNN_EP_COMPILER_PLUGINS (see cmake/HipEpPlugins.cmake).
hipdnn_ep_compiler_plugin_register(PLUGIN_ID sample TARGET hip_ep_sample_plugin)
