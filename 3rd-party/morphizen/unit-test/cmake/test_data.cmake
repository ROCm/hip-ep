##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(RESNET_50_ONNX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data/pt_resnet50.onnx")
if(NOT EXISTS ${RESNET_50_ONNX_PATH})
  message(FATAL_ERROR "Test data file not found: ${RESNET_50_ONNX_PATH}. Please ensure Git LFS is installed and run 'git lfs pull'.")
endif()
# Create a target that depends on the local file
add_custom_target(tgt_pt_resnet50.onnx DEPENDS ${RESNET_50_ONNX_PATH})

set(RESNET_50_MLIR_PATH "${CMAKE_SOURCE_DIR}/ort-bridge/test/src/pt_resnet50.onnx.mlir")
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
