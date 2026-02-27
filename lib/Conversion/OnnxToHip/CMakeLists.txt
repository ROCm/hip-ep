if(ENABLE_ONNX_FRONTEND)
  add_library(OnnxToHip STATIC
    OnnxToHip.cpp
  )

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
    MLIRTransforms
  )
endif()
