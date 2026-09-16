##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_executable(hip-rocmlir-compiler hip-rocmlir-compiler.cpp)

add_dependencies(hip-rocmlir-compiler
  HipDialectIncGen
  HipsrDialectIncGen
  OnnxDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

# The rocMLIR high-level pipeline is invoked at runtime through
# librockCompiler.so via dlopen/dlsym (it embeds its own LLVM/MLIR copy, so it
# must not be link-time coupled). Hence only libdl is needed for the .so hand-
# off; everything else is this executable's own MLIR for the head passes.
target_link_libraries(hip-rocmlir-compiler PRIVATE
  HipSupport
  LibHipCompiler
  HipDialectIR
  HipsrDialectIR
  OnnxDialectIR
  HipDialectTransforms
  HipToTosa
  OnnxToHip
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRFuncTransforms
  ${CMAKE_DL_LIBS}
  ${llvm_libs}
)

target_include_directories(hip-rocmlir-compiler PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/schemas
)

install(TARGETS hip-rocmlir-compiler
  RUNTIME DESTINATION bin
)
