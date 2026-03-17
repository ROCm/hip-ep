##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building HIP Dialect IR Library")

# TableGen targets (HipDialectIncGen, HipTypesIncGen, HipOpsIncGen) are
# defined in include/hip/Dialect/IR/CMakeLists.txt (via add_subdirectory(include/hip)).

# Create HIP dialect IR library
add_library(HipDialectIR STATIC
  HipDialect.cpp
  HipBufferize.cpp
)

# Add dependencies on TableGen targets
add_dependencies(HipDialectIR
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

# Link against MLIR libraries
target_link_libraries(HipDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  ${llvm_libs}
)

# Add include directories
target_include_directories(HipDialectIR PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Set folder for IDE organization
set_target_properties(HipDialectIR PROPERTIES FOLDER "mlir/Dialect")
