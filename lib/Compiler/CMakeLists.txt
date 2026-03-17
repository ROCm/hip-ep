##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Library")

# Add subdirectories
add_subdirectory(Passes)
add_subdirectory(Pipeline)

# Create compiler library
add_library(LibHipCompiler STATIC
  Pipeline.cpp
  CompilerDriver.cpp
)

# Link against required libraries
target_link_libraries(LibHipCompiler PUBLIC
  LibHipCompilerPasses
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
  OMONNXOps
  HipSchemasLib
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

target_link_libraries(LibHipCompiler PUBLIC ${llvm_libs})

# Add include directories
target_include_directories(LibHipCompiler PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include  # For generated TableGen files
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_BINARY_DIR}/proto  # For generated FlatBuffers headers
  ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir
  ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir
)

# Configure custom kernels library path for CompilerDriver
# When building with real runtime, tell CompilerDriver where to find the
# installed hip_custom_kernels.lib at model-compile time.
if(NOT BUILD_MOCK_RUNTIME AND CMAKE_INSTALL_PREFIX)
    if(WIN32)
        set(_custom_kernels_lib "${CMAKE_INSTALL_PREFIX}/lib/hip_custom_kernels.lib")
    else()
        set(_custom_kernels_lib "${CMAKE_INSTALL_PREFIX}/lib/libhip_custom_kernels.a")
    endif()
    target_compile_definitions(LibHipCompiler PRIVATE
        "HIP_CUSTOM_KERNELS_LIB_PATH=\"${_custom_kernels_lib}\""
    )
    message(STATUS "Custom kernels lib path for CompilerDriver: ${_custom_kernels_lib}")
endif()

# Set folder for IDE organization
set_target_properties(LibHipCompiler PROPERTIES FOLDER "mlir/Compiler")
