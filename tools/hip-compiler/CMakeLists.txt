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

# Plugins are linked STATICALLY (see cmake/HipEpPlugins.cmake), so no symbol
# export is needed -- a statically-linked plugin shares this binary's MLIR
# instance by construction. Selected plugin ids are pulled in transitively
# through LibHipCompiler.

# EXPORT HipDnnEpTargets: emit an IMPORTED HipDnnEp::hip-compiler target in the
# installed package config so downstream plugins can locate the compiler via
# find_package(HipDnnEp). install(EXPORT HipDnnEpTargets) is in the top-level
# CMakeLists after add_subdirectory(tools).
install(TARGETS hip-compiler EXPORT HipDnnEpTargets RUNTIME DESTINATION bin)
