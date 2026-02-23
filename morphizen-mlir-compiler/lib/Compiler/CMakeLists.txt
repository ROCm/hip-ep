##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Library")

# Add subdirectories
add_subdirectory(Passes)
add_subdirectory(Pipeline)

# Create compiler library
add_library(MorphizenCompiler STATIC
  Pipeline.cpp
  CompilerDriver.cpp
)

# Link against required libraries
target_link_libraries(MorphizenCompiler PUBLIC
  MorphizenCompilerPasses
  MorphizenConversion
  MorphizenHipDialectIR
  MorphizenHipTransforms
  MorphizenTargetLLVM
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
  MLIRParser
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  OMONNXOps
  MorphizenProto
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

target_link_libraries(MorphizenCompiler PUBLIC ${llvm_libs})

# Add include directories
target_include_directories(MorphizenCompiler PUBLIC
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/include
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/include  # For generated TableGen files
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/lib
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/proto  # For compilation_options.pb.h
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/3rd-party/onnx-mlir
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/3rd-party/onnx-mlir
)

# Set folder for IDE organization
set_target_properties(MorphizenCompiler PROPERTIES FOLDER "mlir/Compiler")
