set(RESNET_50_ONNX_PATH "${CMAKE_CURRENT_BINARY_DIR}/pt_resnet50.onnx")
vaip_add_remote_target(
  TARGET tgt_pt_resnet50.onnx
  FILE ${RESNET_50_ONNX_PATH}
  URL https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/vaip_model/pt_resnet50.onnx
  EXPECTED_MD5 3f97582a85b8ae1be41127f326bbf9b2
)
set_target_properties(tgt_pt_resnet50.onnx PROPERTIES FOLDER morphizen/unit-tests)

morphizen_add_python_target (
  TARGET tgt_test_constant_initializer.onnx
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_constant_initializer.onnx
  SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/vaip/test_constant_initializer.py
  ARGS ${CMAKE_CURRENT_BINARY_DIR}/test_constant_initializer.onnx
  FOLDER "morphizen/unit-tests"
)
morphizen_add_python_target (
  TARGET tgt_sample.onnx
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sample.onnx
  SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/vaip/create_sample_onnx_model.py
  ARGS ${CMAKE_CURRENT_BINARY_DIR}/sample.onnx
  FOLDER "morphizen/unit-tests"
)
morphizen_add_python_target (
  TARGET tgt_test_custom_op.onnx
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_costom_op.onnx
  SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/vaip/test_custom_op.py
  ARGS $${CMAKE_CURRENT_BINARY_DIR}/test_costom_op.onnx
  FOLDER "morphizen/unit-tests"
)
