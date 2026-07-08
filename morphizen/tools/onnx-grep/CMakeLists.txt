##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.5)
project(
  morphizen-onnx-grep
  VERSION 1.0.0
  LANGUAGES C CXX)
find_package(glog REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
set(EXEC_NAME morphizen-onnx-grep)
add_executable(${EXEC_NAME} onnx_grep.cpp ${ryzenai_version_rc_file})
target_include_directories(${EXEC_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
find_package(onnxruntime REQUIRED)
target_link_libraries(${EXEC_NAME} PRIVATE
    morphizen-core-static
    onnxruntime::onnxruntime glog::glog Boost::program_options)

set(ARGS
    "-f ${CMAKE_CURRENT_BINARY_DIR}/../unit-test/pt_resnet50.onnx")
string(APPEND ARGS " -p ${CMAKE_CURRENT_SOURCE_DIR}/../sample_pattern/conv_relu_maxpool.json")
string(APPEND ARGS " -n 135")

set_target_properties(${EXEC_NAME} PROPERTIES
    FOLDER morphizen/tools
    VS_DEBUGGER_COMMAND_ARGUMENTS "${ARGS}"
    VS_DEBUGGER_ENVIRONMENT "PATH=${CMAKE_INSTALL_PREFIX}/bin"
)
target_link_options(${EXEC_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MORPHIZEN_LINKER_OPTIONS}>")
install(TARGETS ${EXEC_NAME} DESTINATION bin)
