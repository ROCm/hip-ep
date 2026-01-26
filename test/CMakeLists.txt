# Option to disable test_onnx_runner compilation
option(BUILD_TEST_ONNX_RUNNER "Build test_onnx_runner executable" OFF)

if(BUILD_TEST_ONNX_RUNNER)
  message(STATUS "Building test_onnx_runner: ENABLED")
else()
  message(STATUS "Building test_onnx_runner: DISABLED")
endif()

# Option to build test_classification executable
option(BUILD_TEST_CLASSIFICATION "Build test_classification executable" ON)

if(BUILD_TEST_CLASSIFICATION)
  message(STATUS "Building test_classification: ENABLED")
else()
  message(STATUS "Building test_classification: DISABLED")
endif()

if(BUILD_TEST_ONNX_RUNNER)

if(NOT TARGET glog::glog)
    find_package(glog CONFIG REQUIRED)
endif()
add_executable(test_onnx_runner
  src/test_onnx_runner.cpp
  src/util/env_config.hpp
  src/util/getenv.c
  src/util/getenv.cpp
  src/util/getopt.c
  src/util/getopt.h
  src/util/parse_value.hpp
)
find_package(onnxruntime CONFIG REQUIRED)
target_link_libraries(test_onnx_runner PRIVATE
  onnxruntime::onnxruntime
  glog::glog)
target_compile_features(test_onnx_runner PRIVATE cxx_std_17)
install(TARGETS test_onnx_runner
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

# Download the test model

if(MSVC)
  add_custom_command(
    TARGET test_onnx_runner
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E echo "COPYING ..."
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_shared.dll
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_vitisai.dll
      $<TARGET_FILE_DIR:test_onnx_runner>
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_shared.dll
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_vitisai.dll
      $<TARGET_FILE_DIR:test_onnx_runner>
  )
endif(MSVC)

set(ENVS "")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_HIPDNN=1")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_PLUGIN=1")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_TAR_CACHE=1")
string(APPEND ENVS "\nXLNX_ENABLE_EP_SHARED_CONTEXT=1")
string(APPEND ENVS "\nVITISAI_EP_JSON_CONFIG=${CMAKE_SOURCE_DIR}/etc/vaip_config.json")
string(APPEND ENVS "\nXLNX_USE_CACHE_DIR=C:/Temp/")
string(APPEND ENVS "\nXLNX_USE_CACHE_KEY=morphizen-hipdnn")
string(APPEND ENVS "\nCACHE_CONTEXT_EMBEDED_MODE=0")
string(APPEND ENVS "\nXLNX_ENABLE_CACHE_CONTEXT=1")
string(APPEND ENVS "\nXLNX_ENABLE_CACHE=0")
string(APPEND ENVS "\nDEBUG_LOG_LEVEL=info")
string(APPEND ENVS "\nXLNX_ONNX_EP_VERBOSE=2")
string(APPEND ENVS "\nMORPHIZEN_DEBUG_DEINITIALIZE=1")
string(APPEND ENVS "\nDEBUG_VAIP_PASS=1")
set_target_properties(test_onnx_runner
	PROPERTIES
	FOLDER "hipdnn"
      VS_DEBUGGER_COMMAND_ARGUMENTS "${morphizen_BINARY_DIR}/unit-test/pt_resnet50.onnx"
      VS_DEBUGGER_ENVIRONMENT "${ENVS}"
  )
add_dependencies(test_onnx_runner ${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET})

add_test(NAME test_generate_ep_context_model
  COMMAND ${CMAKE_COMMAND} -E env
      "DEBUG_LOG_LEVEL=info"
      "XLNX_ONNX_EP_VERBOSE=2"
      "XLNX_ENABLE_CACHE_CONTEXT=1"
      "CACHE_CONTEXT_EMBEDED_MODE=0"
      "XLNX_ENABLE_EP_SHARED_CONTEXT=1"
      "XLNX_ENABLE_CACHE=0"
      "VITISAI_EP_JSON_CONFIG=${CMAKE_SOURCE_DIR}/etc/vaip_config.json"
      $<TARGET_FILE:test_onnx_runner>
      ${morphizen_BINARY_DIR}/unit-test/pt_resnet50.onnx
  WORKING_DIRECTORY ${morphizen_BINARY_DIR}/unit-test
)
set_tests_properties(test_generate_ep_context_model PROPERTIES DEPENDS test_onnx_runner)

add_test(NAME check_generate_ep_context_model
  COMMAND ${CMAKE_COMMAND}
  -DVAR_FILE=${morphizen_BINARY_DIR}/unit-test/pt_resnet50_ctx.onnx
  -P ${CMAKE_CURRENT_LIST_DIR}/cmake/check_file_exists.cmake
)
set_tests_properties(check_generate_ep_context_model PROPERTIES DEPENDS test_generate_ep_context_model)

add_test(NAME test_deploy_ep_context_model
  COMMAND ${CMAKE_COMMAND} -E env
        "DEBUG_LOG_LEVEL=info"
        "XLNX_ONNX_EP_VERBOSE=2"
        "VITISAI_EP_JSON_CONFIG=${CMAKE_SOURCE_DIR}/etc/vaip_config.json"
        $<TARGET_FILE:test_onnx_runner>
        ${morphizen_BINARY_DIR}/unit-test/pt_resnet50_ctx.onnx
  WORKING_DIRECTORY ${morphizen_BINARY_DIR}/unit-test
)
set_tests_properties(test_deploy_ep_context_model PROPERTIES DEPENDS test_generate_ep_context_model)

endif() # BUILD_TEST_ONNX_RUNNER

# Build test_classification executable
if(BUILD_TEST_CLASSIFICATION)

if(NOT TARGET glog::glog)
    find_package(glog CONFIG REQUIRED)
endif()

add_executable(test_classification
  src/test_classification.cpp
  src/util/env_config.hpp
  src/util/getenv.c
  src/util/getenv.cpp
  src/util/getopt.c
  src/util/getopt.h
  src/util/parse_value.hpp
)

find_package(onnxruntime CONFIG REQUIRED)

target_link_libraries(test_classification PRIVATE
  onnxruntime::onnxruntime
  glog::glog)

target_compile_features(test_classification PRIVATE cxx_std_17)

install(TARGETS test_classification
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

if(MSVC)
  add_custom_command(
    TARGET test_classification
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E echo "COPYING onnxruntime DLLs..."
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_shared.dll
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_vitisai.dll
      $<TARGET_FILE_DIR:test_classification>
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_shared.dll
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_vitisai.dll
      $<TARGET_FILE_DIR:test_classification>
  )
endif(MSVC)

set(CLASSIFICATION_ENVS "")
string(APPEND CLASSIFICATION_ENVS "\nUSE_ORT_API_2_0=1")
string(APPEND CLASSIFICATION_ENVS "\nMORPHIZEN_DEBUG_HIPDNN=1")
string(APPEND CLASSIFICATION_ENVS "\nMORPHIZEN_DEBUG_PLUGIN=1")
string(APPEND CLASSIFICATION_ENVS "\nMORPHIZEN_DEBUG_TAR_CACHE=1")
string(APPEND CLASSIFICATION_ENVS "\nXLNX_ENABLE_EP_SHARED_CONTEXT=1")
string(APPEND CLASSIFICATION_ENVS "\nVITISAI_EP_JSON_CONFIG=${CMAKE_SOURCE_DIR}/etc/vaip_config.json")
string(APPEND CLASSIFICATION_ENVS "\nXLNX_USE_CACHE_DIR=C:/Temp/")
string(APPEND CLASSIFICATION_ENVS "\nXLNX_USE_CACHE_KEY=morphizen-hipdnn")
string(APPEND CLASSIFICATION_ENVS "\nCACHE_CONTEXT_EMBEDED_MODE=0")
string(APPEND CLASSIFICATION_ENVS "\nXLNX_ENABLE_CACHE_CONTEXT=1")
string(APPEND CLASSIFICATION_ENVS "\nXLNX_ENABLE_CACHE=0")
string(APPEND CLASSIFICATION_ENVS "\nDEBUG_LOG_LEVEL=info")
string(APPEND CLASSIFICATION_ENVS "\nXLNX_ONNX_EP_VERBOSE=2")
string(APPEND CLASSIFICATION_ENVS "\nMORPHIZEN_DEBUG_DEINITIALIZE=1")
string(APPEND CLASSIFICATION_ENVS "\nDEBUG_VAIP_PASS=1")

set_target_properties(test_classification
  PROPERTIES
  FOLDER "hipdnn"
  VS_DEBUGGER_COMMAND_ARGUMENTS "${CMAKE_SOURCE_DIR}/data/pt_resnet50.onnx ${CMAKE_SOURCE_DIR}/data/resnet50.jpg"
  VS_DEBUGGER_ENVIRONMENT "${CLASSIFICATION_ENVS}"
)

add_dependencies(test_classification ${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET})

# Add CTest for classification
add_test(NAME test_classification_resnet50
  COMMAND ${CMAKE_COMMAND} -E env
      "USE_ORT_API_2_0=1"
      "DEBUG_LOG_LEVEL=info"
      "XLNX_ONNX_EP_VERBOSE=2"
      "XLNX_ENABLE_CACHE_CONTEXT=1"
      "CACHE_CONTEXT_EMBEDED_MODE=0"
      "XLNX_ENABLE_EP_SHARED_CONTEXT=1"
      "XLNX_ENABLE_CACHE=0"
      "VITISAI_EP_JSON_CONFIG=${CMAKE_SOURCE_DIR}/etc/vaip_config.json"
      $<TARGET_FILE:test_classification>
      ${CMAKE_SOURCE_DIR}/data/pt_resnet50.onnx
      ${CMAKE_SOURCE_DIR}/data/resnet50.jpg
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_classification_resnet50 PROPERTIES DEPENDS test_classification)

endif() # BUILD_TEST_CLASSIFICATION
