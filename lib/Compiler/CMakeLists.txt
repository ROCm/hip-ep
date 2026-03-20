##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Library")

# Create compiler library
add_library(LibHipCompiler STATIC
  CompilerDriver.cpp
)

# Link against required libraries
target_link_libraries(LibHipCompiler PUBLIC
  HipConversion
  HipDialectIR
  HipDialectTransforms
  HipTargetLLVM
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRParser
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  HipSchemasLib
)

# Link LLVM components
llvm_map_components_to_libnames(compiler_llvm_libs
  Support
  Core
  IRReader
  Passes
  Target
  X86CodeGen
  X86AsmParser
  X86Desc
  X86Info
)

target_link_libraries(LibHipCompiler PUBLIC ${compiler_llvm_libs})

target_include_directories(LibHipCompiler PUBLIC
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_BINARY_DIR}/schemas
  ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir
  ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir
)

# Set folder for IDE organization
set_target_properties(LibHipCompiler PROPERTIES FOLDER "mlir/Compiler")
