# Option to build test_classification executable
option(BUILD_TEST_CLASSIFICATION "Build test_classification executable" ON)

if(BUILD_TEST_CLASSIFICATION)
  message(STATUS "Building test_classification: ENABLED")
else()
  message(STATUS "Building test_classification: DISABLED")
endif()

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
      $<TARGET_FILE_DIR:test_classification>
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:onnxruntime::onnxruntime>/onnxruntime_providers_shared.dll
      $<TARGET_FILE_DIR:test_classification>
  )
endif(MSVC)

set(CLASSIFICATION_ENVS "")
string(APPEND CLASSIFICATION_ENVS "\nMORPHIZEN_DEBUG_HIPDNN=1")
string(APPEND CLASSIFICATION_ENVS "\nMORPHIZEN_DEBUG_PLUGIN=1")
string(APPEND CLASSIFICATION_ENVS "\nXLNX_ENABLE_EP_SHARED_CONTEXT=1")
string(APPEND CLASSIFICATION_ENVS "\nMORPHIZEN_EP_JSON_CONFIG=${CMAKE_SOURCE_DIR}/etc/vaip_config.json")
string(APPEND CLASSIFICATION_ENVS "\nXLNX_USE_CACHE_DIR=C:/Temp/")
string(APPEND CLASSIFICATION_ENVS "\nXLNX_USE_CACHE_KEY=onnx-hipdnn-ep")
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

add_dependencies(test_classification ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})

# Add CTest for classification
add_test(NAME test_classification_resnet50
  COMMAND ${CMAKE_COMMAND} -E env
      "DEBUG_LOG_LEVEL=info"
      "XLNX_ONNX_EP_VERBOSE=2"
      "XLNX_ENABLE_CACHE_CONTEXT=1"
      "CACHE_CONTEXT_EMBEDED_MODE=0"
      "XLNX_ENABLE_EP_SHARED_CONTEXT=1"
      "XLNX_ENABLE_CACHE=0"
      "MORPHIZEN_EP_JSON_CONFIG=${CMAKE_SOURCE_DIR}/etc/vaip_config.json"
      $<TARGET_FILE:test_classification>
      ${CMAKE_SOURCE_DIR}/data/pt_resnet50.onnx
      ${CMAKE_SOURCE_DIR}/data/resnet50.jpg
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(test_classification_resnet50 PROPERTIES DEPENDS test_classification)

endif() # BUILD_TEST_CLASSIFICATION
