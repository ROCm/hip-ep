##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipToLLVM STATIC
  HipToLLVM.cpp
)

add_dependencies(HipToLLVM
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(HipToLLVM PUBLIC
  HipDialectIR
  MLIRPass
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRMemRefDialect
  MLIRTransforms
)
