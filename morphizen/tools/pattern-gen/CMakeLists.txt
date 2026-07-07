##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.5)
project(
  morphizen-pattern-gen
  VERSION 1.0.0
  LANGUAGES C CXX)
find_package(glog REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
set(EXEC_NAME morphizen-pattern-gen)
add_executable(${EXEC_NAME} onnx_pattern_gen.cpp ${ryzenai_version_rc_file})
target_include_directories(${EXEC_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
find_package(onnxruntime REQUIRED)
target_link_libraries(${EXEC_NAME} PRIVATE
    morphizen-core-static
    onnxruntime::onnxruntime glog::glog Boost::program_options)
set(ARGS
	"-f ${CMAKE_CURRENT_BINARY_DIR}/../unit-test/pt_resnet50.onnx")
string(APPEND ARGS " -m ${CMAKE_CURRENT_BINARY_DIR}/conv_relu_maxpool.mmd")
string(APPEND ARGS " -i 111")
string(APPEND ARGS " -o 135")
string(APPEND ARGS " -j ${CMAKE_CURRENT_BINARY_DIR}/conv_relu_maxpool.json")
set_target_properties(${EXEC_NAME} PROPERTIES
    FOLDER morphizen/tools
    VS_DEBUGGER_COMMAND_ARGUMENTS "${ARGS}"
)
target_link_options(${EXEC_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MORPHIZEN_LINKER_OPTIONS}>")
install(TARGETS ${EXEC_NAME} DESTINATION bin)

if(morphizen_ENABLE_UNIT_TEST)
  add_test(NAME ${EXEC_NAME}_test
           COMMAND $<TARGET_FILE:${EXEC_NAME}>
            -f ${CMAKE_CURRENT_BINARY_DIR}/../unit-test/pt_resnet50.onnx
            -m ${CMAKE_CURRENT_BINARY_DIR}/conv_relu_maxpool.mmd
            -i 111
            -o 135
            -j ${CMAKE_CURRENT_BINARY_DIR}/conv_relu_maxpool.json)

  add_test(NAME ${EXEC_NAME}_onnx-grep_test
           COMMAND $<TARGET_FILE:morphizen-onnx-grep>
            -f ${CMAKE_CURRENT_BINARY_DIR}/../unit-test/pt_resnet50.onnx
            -p ${CMAKE_CURRENT_BINARY_DIR}/conv_relu_maxpool.json
            -n 135)

  set_tests_properties(${EXEC_NAME}_onnx-grep_test PROPERTIES
                       DEPENDS ${EXEC_NAME}_test)
endif()
