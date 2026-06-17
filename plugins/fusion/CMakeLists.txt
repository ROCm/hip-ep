##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
#
# Out-of-tree fusion plugin for hip-mlir-opt.
#
# Builds TWO dlopen-able MODULE libraries that contribute a real hip.* op
# (hip.fused_mul_add) plus a named pass (--hip-fuse-mul-add) at load time, with
# no rebuild of hip-mlir-opt beyond the one-time HipDialect::registerPluginOps
# hook:
#
#   libhip_fusion_plugin.so        — attaches the bufferization external model.
#   libhip_fusion_plugin_nobuf.so  — omits it (compiled with
#                                    HIPPOC_SKIP_BUFFERIZE_ATTACH), to
#                                    demonstrate the §4 / R-1 failure mode.
#
# The modules deliberately do NOT static-link MLIR/LLVM or HipDialectIR: MLIR
# core symbols and the in-tree hip.* op symbols (AddOp/MulOp/...) are resolved
# from the host hip-mlir-opt process at dlopen time. Static-linking those would
# duplicate TypeID/registry singletons and break op identity across the .so
# boundary.
# ==============================================================================

# --- TableGen: plugin op (hip.fused_mul_add) ---------------------------------
set(LLVM_TARGET_DEFINITIONS FusedMulAddOps.td)
mlir_tablegen(FusedMulAddOps.h.inc -gen-op-decls)
mlir_tablegen(FusedMulAddOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(HipFusionPluginOpsIncGen)

# --- TableGen: plugin pass (--hip-fuse-mul-add) ------------------------------
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HipFusion)
add_public_tablegen_target(HipFusionPluginPassesIncGen)

# Generated *.inc land in ${CMAKE_CURRENT_BINARY_DIR}; the sources include them
# as "plugins/fusion/<file>.h.inc", so the include root is ${CMAKE_BINARY_DIR}.
set(_plugin_srcs
  PluginMain.cpp
  PluginOps.cpp
  FuseMulAddPass.cpp
  FusedMulAddLowering.cpp   # Touchpoint E: HipToLLVM lowering pattern
  PipelineSlot.cpp          # Touchpoint C: pipeline anchor registry
)

# Build both variants from the same sources.
foreach(_variant IN ITEMS "" "_nobuf")
  set(_tgt hip_fusion_plugin${_variant})

  # On Windows export mlirGetDialectPluginInfo / mlirGetPassPluginInfo via a
  # .def file — __declspec(dllexport) would conflict with the LLVM_ATTRIBUTE_WEAK
  # forward-declaration in DialectPlugin.h / PassPlugin.h headers.
  if(WIN32)
    set(_def_file "${CMAKE_CURRENT_SOURCE_DIR}/hip_fusion_plugin.def")
  else()
    set(_def_file "")
  endif()

  add_library(${_tgt} MODULE ${_plugin_srcs} ${_def_file})

  add_dependencies(${_tgt}
    HipDialectIncGen
    HipTypesIncGen
    HipOpsIncGen
    HipFusionPluginOpsIncGen
    HipFusionPluginPassesIncGen
  )

  target_include_directories(${_tgt} PRIVATE
    ${CMAKE_SOURCE_DIR}                  # plugin sources via "plugins/fusion/..."
    ${CMAKE_SOURCE_DIR}/include          # in-tree hip/ headers
    ${CMAKE_BINARY_DIR}/include          # generated hip/...h.inc
    ${CMAKE_BINARY_DIR}                  # generated plugins/fusion/...h.inc
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
  )

  # Match the RTTI-off ABI of the dist LLVM; MODULE targets are already PIC.
  target_compile_options(${_tgt} PRIVATE -fno-rtti)

  # Link MLIR so the plugin's MLIR-core symbols and TypeIDs match the host.
  # Linux: link libMLIR.so (the dylib) — required with hidden-visibility LLVM.
  # Windows (MSVC): link static MLIR components — the EXE exports all symbols
  # by default, so TypeIDs from the EXE are shared with the plugin .dll.
  # In both cases do NOT link HipDialectIR: in-tree hip.* op TypeIDs resolve
  # from the host process at load time via export_executable_symbols_for_plugins.
  # On Linux link libMLIR.so (dylib) so TypeIDs are shared with the host.
  # On Windows the EXE exports all symbols by default; the plugin DLL links
  # against the EXE's import lib so unresolved HipDialectIR / MLIR symbols
  # (AddOp, MulOp, ContextType TypeIDs, createConvertHipToLLVMPass, etc.)
  # all resolve from the host at load time — no static copy needed.
  # On Linux link libMLIR.so (dylib) so TypeIDs are shared with the host.
  # On Windows link the EXE import lib (for HipDialect/HipToLLVM symbols that
  # export_executable_symbols_for_plugins makes visible) PLUS the static LLVM/
  # MLIR component libs (for LLVM core: SmallVector, report_fatal_error, etc.).
  find_library(_mlir_dylib_plugin NAMES MLIR
    HINTS ${LLVM_LIBRARY_DIR} ${MLIR_LIBRARY_DIR}
    NO_DEFAULT_PATH)
  if(_mlir_dylib_plugin OR NOT WIN32)
    target_link_libraries(${_tgt} PRIVATE MLIR)
  else()
    # Windows: link ONLY against the EXE import library. This ensures the plugin
    # resolves ALL symbols (HipDialectIR, MLIR, LLVM) from the single EXE image,
    # avoiding the duplicate-TypeID crash that occurs when the plugin brings in
    # its own copy of static MLIR/LLVM libs alongside the EXE's copies.
    # export_executable_symbols_for_plugins() on the EXE makes all its symbols
    # available for import.
    # EXE import lib: resolves all HipDialect, MLIR, and LLVM singleton symbols
    # (TypeIDs, dialect registries, pass managers) from the single EXE image,
    # preventing duplicate-singleton crashes.
    # LLVMSupport static lib: provides non-singleton LLVM runtime helpers
    # (report_fatal_error, SmallVectorBase, allocate_buffer) that are not
    # exported from the EXE because they are inlined or template-instantiated.
    target_link_libraries(${_tgt} PRIVATE
      "$<TARGET_LINKER_FILE:hip-mlir-opt>"
      LLVMSupport
    )
    add_dependencies(${_tgt} hip-mlir-opt)
  endif()

  set_target_properties(${_tgt} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
  )

  if(_variant STREQUAL "_nobuf")
    target_compile_definitions(${_tgt} PRIVATE HIPPOC_SKIP_BUFFERIZE_ATTACH)
  endif()
endforeach()

# ==============================================================================
# Touchpoints F + G: runtime wrapper + gfx1103 HIP kernel
# ==============================================================================
# Builds a shared library containing:
#   F. wrap_fused_mul_add() — C-ABI runtime symbol called by generated LLVM IR
#   G. fused_mul_add_kernel — gfx1103 HIP device kernel (via hipcc)
#
# The plugin registers this library via addLibrary/addLibraryPath in a
# production ABI. For the PoC, it is linked with hip-mlir-opt for e2e testing.
# ==============================================================================

# Compile the HIP kernel (gfx1103) to an object file via hipcc.
# Only attempt on Linux (hipcc is a shell script); on Windows the .hip source
# is documented for reference and the runtime test is run on Linux/ETX.
if(NOT WIN32 AND THEROCK_DIST)
  find_program(HIPCC_EXECUTABLE hipcc
    HINTS "${THEROCK_DIST}/bin"
    NO_DEFAULT_PATH
  )
  if(HIPCC_EXECUTABLE)
    set(_kernel_src ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fused_mul_add_kernel.hip)
    set(_kernel_obj ${CMAKE_CURRENT_BINARY_DIR}/fused_mul_add_kernel.hip.o)
    add_custom_command(
      OUTPUT  ${_kernel_obj}
      COMMAND ${HIPCC_EXECUTABLE}
              --offload-arch=gfx1103
              -fPIC -O2 -c
              -o ${_kernel_obj}
              ${_kernel_src}
      DEPENDS ${_kernel_src}
      COMMENT "Compiling fused_mul_add_kernel.hip for gfx1103"
    )
    set(_kernel_obj_dep ${_kernel_obj})
  else()
    message(STATUS "[plugin/fusion] hipcc not found — skipping kernel compile")
    set(_kernel_obj_dep "")
  endif()
else()
  set(_kernel_obj_dep "")
endif()

add_library(hip_fusion_runtime SHARED
  runtime/wrap_fused_mul_add.cpp
  ${_kernel_obj_dep}
)

target_include_directories(hip_fusion_runtime PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)

set_target_properties(hip_fusion_runtime PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
  OUTPUT_NAME "hip_fusion_runtime"
)

if(THEROCK_DIST AND NOT WIN32)
  target_link_directories(hip_fusion_runtime PRIVATE "${THEROCK_DIST}/lib")
  target_link_libraries(hip_fusion_runtime PRIVATE amdhip64)
endif()
