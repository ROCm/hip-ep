##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_executable(hip-compiler hip-compiler.cpp)
set_target_properties(hip-compiler PROPERTIES PDB_NAME "hip-compiler-cli")

add_dependencies(hip-compiler
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(hip-compiler PRIVATE
  LibHipCompiler
  LLVMSupport
  HipSupport
)

target_include_directories(hip-compiler PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/schemas
)

# Export this executable's statically-linked MLIR symbols so a plugin loaded
# via HIP_EP_PLUGINS (dlopen'd into the hip-compiler process) binds to THIS
# host's single MLIR pass registry. Without it, the plugin's
# mlir::PassRegistration writes a private registry the host never reads (the
# in-tree XFAIL). This is the exact mechanism mlir-opt uses
# (mlir/tools/mlir-opt/CMakeLists.txt). Gated by HIPDNN_ENABLE_PLUGINS, which
# sets LLVM_ENABLE_PLUGINS so this call takes effect.
if(HIPDNN_ENABLE_PLUGINS)
  export_executable_symbols_for_plugins(hip-compiler)
endif()

# EXPORT HipDnnEpTargets: emit an IMPORTED HipDnnEp::hip-compiler target in the
# installed package config so downstream plugins can locate the compiler via
# find_package(HipDnnEp). install(EXPORT HipDnnEpTargets) is in the top-level
# CMakeLists after add_subdirectory(tools).
install(TARGETS hip-compiler EXPORT HipDnnEpTargets RUNTIME DESTINATION bin)
