##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.5)
project(morphizen-tar VERSION 1.0.0 LANGUAGES C CXX)
find_package(glog REQUIRED)
set(EXEC_NAME morphizen-tar)
add_executable(${EXEC_NAME} src/tar_main.cpp)
target_include_directories(${EXEC_NAME}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(${EXEC_NAME} PRIVATE morphizen-core-static glog::glog)
set_target_properties(${EXEC_NAME} PROPERTIES
	FOLDER morphizen/tools
	VS_DEBUGGER_COMMAND_ARGUMENTS "-l ${CMAKE_CURRENT_BINARY_DIR}/../unit-test/sample.src.tar"
)
install(TARGETS ${EXEC_NAME} DESTINATION bin)
