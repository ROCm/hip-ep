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

add_custom_target(tgt_sample_src_tar
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/sample_src_tar
  COMMAND ${CMAKE_COMMAND} -E echo hello >${CMAKE_CURRENT_BINARY_DIR}/sample_src_tar/hello.txt
  COMMAND ${CMAKE_COMMAND} -E echo "tar file" >${CMAKE_CURRENT_BINARY_DIR}/sample_src_tar/tar_file.txt
  COMMAND ${CMAKE_COMMAND} -E tar cvf ${CMAKE_CURRENT_BINARY_DIR}/sample.src.tar
  sample_src_tar/hello.txt
  sample_src_tar/tar_file.txt
)
set_target_properties(tgt_sample_src_tar PROPERTIES FOLDER morphizen/unit-tests/data)
