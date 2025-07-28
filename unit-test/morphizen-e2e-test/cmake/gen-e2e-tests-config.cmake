##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(MORPHIZEN_E2E_TESTS_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/morphizen_e2e_tests_config.json)
add_custom_command(
  OUTPUT ${MORPHIZEN_E2E_TESTS_CONFIG_FILE}
  COMMAND
    ${CMAKE_COMMAND} -E echo "generator env config json file ..."
    $<TARGET_FILE:Python3::Interpreter> ${CMAKE_CURRENT_LIST_DIR}/../etc/gen_e2e_tests_config.py
    ${MORPHIZEN_E2E_TESTS_CONFIG_FILE}
  COMMAND
    $<TARGET_FILE:Python3::Interpreter> ${CMAKE_CURRENT_LIST_DIR}/../etc/gen_e2e_tests_config.py
    ${MORPHIZEN_E2E_TESTS_CONFIG_FILE}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/../etc/gen_e2e_tests_config.py
)
add_custom_target(morphizen_e2e_tests_generate_env_config
  DEPENDS ${MORPHIZEN_E2E_TESTS_CONFIG_FILE}
  COMMENT "Generating test env config json file ..."
)
target_sources(morphizen_e2e_tests_generate_env_config
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../etc/gen_e2e_tests_config.py
    ${MORPHIZEN_E2E_TESTS_CONFIG_FILE}
)

set_target_properties(morphizen_e2e_tests_generate_env_config
  PROPERTIES
  FOLDER "morphizen/unit-tests"
)
