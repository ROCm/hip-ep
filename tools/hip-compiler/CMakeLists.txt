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

install(TARGETS hip-compiler RUNTIME DESTINATION bin)
