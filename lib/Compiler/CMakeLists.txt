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

# Configure custom kernels library path for CompilerDriver
# When building with real runtime, tell CompilerDriver where to find the
# installed hip_custom_kernels.lib at model-compile time.
if(NOT BUILD_MOCK_RUNTIME AND TARGET hip_custom_kernels)
    # Provide CompilerDriver with the exact path to the built custom kernels
    # archive so it can inject the library on the model DLL link line without
    # relying on an install step.
    target_compile_definitions(LibHipCompiler PRIVATE
        "HIP_CUSTOM_KERNELS_LIB_PATH=\"$<TARGET_FILE:hip_custom_kernels>\""
    )
    message(STATUS "Custom kernels lib path for CompilerDriver: $<TARGET_FILE:hip_custom_kernels>")
endif()

# hipDNN graph runtime import library path for model.dll linking.
# Uses generator expression to get the actual import library location.
if(TARGET hipdnn_graph_runtime)
    target_compile_definitions(LibHipCompiler PRIVATE
        "HIPDNN_GRAPH_RUNTIME_LIB_PATH=\"$<TARGET_LINKER_FILE:hipdnn_graph_runtime>\""
    )
    target_link_libraries(LibHipCompiler PUBLIC OnnxToHipDNN)
endif()

# Set folder for IDE organization
set_target_properties(LibHipCompiler PROPERTIES FOLDER "mlir/Compiler")
