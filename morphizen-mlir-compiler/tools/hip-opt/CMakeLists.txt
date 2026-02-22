##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Add the executable (links against MorphizenHipDialect static library)
add_executable(hip-opt
  hip-opt.cpp
)

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs Support Core)

# Link against MorphizenHipDialect library and MLIR libraries
target_link_libraries(hip-opt PRIVATE
  MorphizenHipDialect
  MLIROptLib
  MLIRIR
  MLIRPass
  MLIRSupport
  ${llvm_libs}
  OMONNXOps  # ONNX dialect from our 3rd-party/onnx-mlir
)

# Install
install(TARGETS hip-opt
    RUNTIME DESTINATION bin
)
