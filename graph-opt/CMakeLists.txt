##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.5)
project(morphizen-graph-opt VERSION 1.0.0 LANGUAGES C CXX)

find_package(glog REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(onnxruntime REQUIRED)
set(EXEC_NAME morphizen-graph-opt)
add_executable(${EXEC_NAME} graph-opt.cpp ${ryzenai_version_rc_file})
target_include_directories(${EXEC_NAME}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(${EXEC_NAME} PRIVATE
  ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
	glog::glog
  onnxruntime::onnxruntime
  Boost::program_options)
target_link_options(${EXEC_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MORPHIZEN_LINKER_OPTIONS}>")
set_target_properties(${EXEC_NAME} PROPERTIES FOLDER morphizen/tools)
install(TARGETS ${EXEC_NAME} DESTINATION bin)

# if(morphizen_ENABLE_UNIT_TEST)
#   add_test(NAME ${EXEC_NAME}_test COMMAND $<TARGET_FILE:${EXEC_NAME}> -i ${CMAKE_CURRENT_BINARY_DIR}/../unit-test/pt_resnet50.onnx -p morphizen-pass_init)
# endif()

# originally morphizen-graph-opt is callled `voe_py_pass`, to minimize
# changes at morphizen side we add an alias here.
add_executable (voe_py_pass ALIAS ${EXEC_NAME})
