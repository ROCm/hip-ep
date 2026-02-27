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
