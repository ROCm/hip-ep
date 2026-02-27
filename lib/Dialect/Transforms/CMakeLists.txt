##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectTransforms STATIC
  LowerAllocs.cpp
  OptimizeMemRefs.cpp
  PoolAllocs.cpp
)

add_dependencies(HipDialectTransforms
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(HipDialectTransforms PUBLIC
  HipDialectIR
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
)
