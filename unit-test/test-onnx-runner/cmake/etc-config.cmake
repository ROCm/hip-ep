##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(TEST_ONNX_RUNNER_ENV_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/env_config.json)
add_custom_command(
  OUTPUT ${TEST_ONNX_RUNNER_ENV_CONFIG_FILE}
  COMMAND
    ${CMAKE_COMMAND} -E echo "generator env config json file ..."
    $<TARGET_FILE:Python3::Interpreter> ${CMAKE_CURRENT_LIST_DIR}/../etc/default_env_config.py
    ${TEST_ONNX_RUNNER_ENV_CONFIG_FILE}
  COMMAND
    $<TARGET_FILE:Python3::Interpreter> ${CMAKE_CURRENT_LIST_DIR}/../etc/default_env_config.py
    ${TEST_ONNX_RUNNER_ENV_CONFIG_FILE}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/../etc/default_env_config.py
)
add_custom_target(test_onnx_runner_generate_env_config
  DEPENDS ${TEST_ONNX_RUNNER_ENV_CONFIG_FILE}
  COMMENT "Generating test env config json file ..."
)
target_sources(test_onnx_runner_generate_env_config
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../etc/default_env_config.py
    ${TEST_ONNX_RUNNER_ENV_CONFIG_FILE}
)

set_target_properties(test_onnx_runner_generate_env_config
  PROPERTIES
  FOLDER "morphizen/unit-tests"
)
