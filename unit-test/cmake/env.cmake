##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
if(WIN32)
  set(PATH_ENV_VAR_NAME "PATH")
else(WIN32)
  set(PATH_ENV_VAR_NAME "LD_LIBRARY_PATH")
endif(WIN32)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/unit_test_env.txt" VAIP_UNIT_TEST_ENV_IN)
string(CONFIGURE "${VAIP_UNIT_TEST_ENV_IN}" VAIP_UNIT_TEST_ENV @ONLY)
string(REPLACE "\n" ";" VAIP_UNIT_TEST_ENV_LIST ${VAIP_UNIT_TEST_ENV})
# message(STATUS "LIST OF ENVIRONMENT VARIABLES in ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_env.txt :")
configure_file("vs.runsettings.in" ${CMAKE_BINARY_DIR}/.runsettings)
set(TEST_ENV "")
foreach(element IN LISTS VAIP_UNIT_TEST_ENV_LIST)
  # message(STATUS "   ENV VAR FOR TESTING:  ${element}")
  list(APPEND TEST_ENV ENVIRONMENT "${element}")
endforeach(element IN LISTS TMP_COMPONENT_CONTENT)
