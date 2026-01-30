##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

if(NOT TARGET glog::glog)
    find_package(glog CONFIG REQUIRED)
endif()
find_package(Boost REQUIRED COMPONENTS program_options)
add_executable(test_onnx_runner
  src/test_onnx_runner.cpp
  src/util/env_config.hpp
  src/util/getenv.c
  src/util/getenv.cpp
  src/util/parse_value.hpp
)
find_package(onnxruntime CONFIG REQUIRED)
target_link_libraries(test_onnx_runner PRIVATE
  onnxruntime::onnxruntime
  glog::glog
  Boost::program_options)
target_compile_features(test_onnx_runner PRIVATE cxx_std_17)
install(TARGETS test_onnx_runner
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

# Download the test model

# Copy onnxruntime DLLs to output directory on Windows
if(MSVC)
  add_custom_command(
    TARGET test_onnx_runner
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E echo "COPYING onnxruntime.dll ..."
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:test_onnx_runner>
  )
endif(MSVC)

set(ENVS "")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_RELU_DQ=1")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_PLUGIN=1")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_TAR_CACHE=1")
string(APPEND ENVS "\nXLNX_ENABLE_EP_SHARED_CONTEXT=1")
string(APPEND ENVS "\nVITISAI_EP_JSON_CONFIG=${CMAKE_SOURCE_DIR}/etc/morphizen_config.json")
string(APPEND ENVS "\nXLNX_USE_CACHE_DIR=C:/Temp/")
string(APPEND ENVS "\nXLNX_USE_CACHE_KEY=morphizen-demo")
string(APPEND ENVS "\nCACHE_CONTEXT_EMBEDED_MODE=0")
string(APPEND ENVS "\nXLNX_ENABLE_CACHE_CONTEXT=1")
string(APPEND ENVS "\nXLNX_ENABLE_CACHE=0")
string(APPEND ENVS "\nDEBUG_LOG_LEVEL=info")
string(APPEND ENVS "\nXLNX_ONNX_EP_VERBOSE=2")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_DEINITIALIZE=1")
string(APPEND ENVS "\nDEBUG_MORPHIZEN_PASS=1")
set_target_properties(test_onnx_runner
	PROPERTIES
	FOLDER "relu-dq"
      VS_DEBUGGER_COMMAND_ARGUMENTS "${morphizen_BINARY_DIR}/unit-test/pt_resnet50.onnx"
      VS_DEBUGGER_ENVIRONMENT "${ENVS}"
  )
add_dependencies(test_onnx_runner ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})

# add_test(NAME test_generate_ep_context_model
#  COMMAND ${CMAKE_COMMAND} -E env
#      "DEBUG_LOG_LEVEL=info"
#      "XLNX_ONNX_EP_VERBOSE=2"
#      "XLNX_ENABLE_CACHE_CONTEXT=1"
#      "CACHE_CONTEXT_EMBEDED_MODE=0"
#      "XLNX_ENABLE_EP_SHARED_CONTEXT=1"
#      "XLNX_ENABLE_CACHE=0"
#      "VITISAI_EP_JSON_CONFIG=${CMAKE_SOURCE_DIR}/etc/morphizen_config.json"
#      $<TARGET_FILE:test_onnx_runner>
#      ${morphizen_BINARY_DIR}/unit-test/pt_resnet50.onnx
#  WORKING_DIRECTORY ${morphizen_BINARY_DIR}/unit-test
# )
# set_tests_properties(test_generate_ep_context_model PROPERTIES DEPENDS test_onnx_runner)

# add_test(NAME check_generate_ep_context_model
# COMMAND ${CMAKE_COMMAND}
# # -DVAR_FILE=${morphizen_BINARY_DIR}/unit-test/pt_resnet50_ctx.onnx
# -P ${CMAKE_CURRENT_LIST_DIR}/cmake/check_file_exists.cmake
# )
# set_tests_properties(check_generate_ep_context_model PROPERTIES DEPENDS test_generate_ep_context_model)
#
# add_test(NAME test_deploy_ep_context_model
# COMMAND ${CMAKE_COMMAND} -E env
# "DEBUG_LOG_LEVEL=info"
# "XLNX_ONNX_EP_VERBOSE=2"
# "VITISAI_EP_JSON_CONFIG=${CMAKE_SOURCE_DIR}/etc/morphizen_config.json"
# $<TARGET_FILE:test_onnx_runner>
# ${morphizen_BINARY_DIR}/unit-test/pt_resnet50_ctx.onnx
# WORKING_DIRECTORY ${morphizen_BINARY_DIR}/unit-test
# )
# set_tests_properties(test_deploy_ep_context_model PROPERTIES DEPENDS test_generate_ep_context_model)
