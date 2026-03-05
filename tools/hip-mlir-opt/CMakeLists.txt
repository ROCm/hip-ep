##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Add the executable
add_executable(hip-opt
  hip-opt.cpp
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

# Link against required libraries
# Note: hip-opt uses UdnaPipeline (lightweight, no LLD dependencies)
# instead of UdnaCompiler (which includes backend and LLD dependencies)
target_link_libraries(hip-opt PRIVATE
  UdnaPipeline  # For Pipeline.cpp (registerUdnaPipeline) - no LLD deps
  UdnaHipDialectIR
  UdnaHipTransforms
  UdnaConversion
  MLIROptLib  # Includes mlir::registerAllPasses() implementation
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms  # For canonicalizer and other standard passes
  MLIRArithTransforms          # arith bufferization models
  MLIRTensorTransforms         # tensor bufferization models
  MLIRBufferizationTransforms  # func_ext bufferization models
  ${llvm_libs}
  OMONNXOps
)

# Add include directories
target_include_directories(hip-opt PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir
  ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir
  ${CMAKE_BINARY_DIR}/proto
)

# Install
install(TARGETS hip-opt
    RUNTIME DESTINATION bin
)
