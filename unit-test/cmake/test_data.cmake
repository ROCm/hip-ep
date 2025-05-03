##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(RESNET_50_ONNX_PATH "${CMAKE_CURRENT_BINARY_DIR}/pt_resnet50.onnx")
vaip_add_remote_target(
  TARGET tgt_pt_resnet50.onnx
  FILE ${RESNET_50_ONNX_PATH}
  URL https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/vaip_model/pt_resnet50.onnx
  EXPECTED_MD5 3f97582a85b8ae1be41127f326bbf9b2
)
set_target_properties(tgt_pt_resnet50.onnx PROPERTIES FOLDER morphizen/unit-tests/data)

morphizen_add_python_target (
  TARGET tgt_test_constant_initializer.onnx
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_constant_initializer.onnx
  SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/vaip/test_constant_initializer.py
  ARGS ${CMAKE_CURRENT_BINARY_DIR}/test_constant_initializer.onnx
  FOLDER "morphizen/unit-tests/data"
)
morphizen_add_python_target (
  TARGET tgt_sample.onnx
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sample.onnx
  SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/vaip/create_sample_onnx_model.py
  ARGS ${CMAKE_CURRENT_BINARY_DIR}/sample.onnx
  FOLDER "morphizen/unit-tests/data"
)
morphizen_add_python_target (
  TARGET tgt_test_custom_op.onnx
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_costom_op.onnx
  SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/vaip/test_custom_op.py
  ARGS ${CMAKE_CURRENT_BINARY_DIR}/test_costom_op.onnx
  FOLDER "morphizen/unit-tests/data"
)

add_custom_target(tgt_sample_src_tar
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/sample_src_tar
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/vaip/test_config.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/vaip/test_tarball.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/sample_src_tar
  COMMAND ${CMAKE_COMMAND} -E tar cvf ${CMAKE_CURRENT_BINARY_DIR}/sample.src.tar
  sample_src_tar/test_config.cpp
  sample_src_tar/test_tarball.cpp
)
set_target_properties(tgt_sample_src_tar PROPERTIES FOLDER morphizen/unit-tests/data)
