##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Library")

# Add subdirectories
add_subdirectory(Passes)
add_subdirectory(Pipeline)

# Create compiler library
add_library(UdnaCompiler STATIC
  Pipeline.cpp
  CompilerDriver.cpp
)

# Link against required libraries
target_link_libraries(UdnaCompiler PUBLIC
  UdnaCompilerPasses
  UdnaConversion
  UdnaHipDialectIR
  UdnaHipTransforms
  UdnaTargetLLVM
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
  OMONNXOps
  UdnaSchemasLib
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs
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

target_link_libraries(UdnaCompiler PUBLIC ${llvm_libs})

# Add include directories
target_include_directories(UdnaCompiler PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include  # For generated TableGen files
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_BINARY_DIR}/proto  # For generated FlatBuffers headers
  ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir
  ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir
)

# Set folder for IDE organization
set_target_properties(UdnaCompiler PROPERTIES FOLDER "mlir/Compiler")
