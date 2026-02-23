##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Add the executable
add_executable(morphizen-opt
  morphizen-opt.cpp
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

# Link against required libraries
# Note: morphizen-opt uses MorphizenPipeline (lightweight, no LLD dependencies)
# instead of MorphizenCompiler (which includes backend and LLD dependencies)
target_link_libraries(morphizen-opt PRIVATE
  MorphizenPipeline  # For Pipeline.cpp (registerMorphizenPipeline) - no LLD deps
  MorphizenHipDialectIR
  MorphizenHipTransforms
  MorphizenConversion
  MLIROptLib  # Includes mlir::registerAllPasses() implementation
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms  # For canonicalizer and other standard passes
  ${llvm_libs}
  OMONNXOps
)

# Add include directories
target_include_directories(morphizen-opt PRIVATE
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/include
  ${CMAKE_SOURCE_DIR}/morphizen-mlir-compiler/3rd-party/onnx-mlir
  ${CMAKE_BINARY_DIR}/morphizen-mlir-compiler/3rd-party/onnx-mlir
)

# Install
install(TARGETS morphizen-opt
    RUNTIME DESTINATION bin
)
