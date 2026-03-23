##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(OnnxToHip STATIC
  OnnxToHip.cpp
)

# Disable C4244 warning (type conversion) on MSVC
# These warnings come from LLVM JSON library usage with std::optional
if(MSVC)
  target_compile_options(OnnxToHip PRIVATE /wd4244)
endif()

add_dependencies(OnnxToHip
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(OnnxToHip PUBLIC
  HipDialectIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms
)
