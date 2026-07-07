##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
function(morphizen_add_remote_target)
  set(options)
  set(oneValueArgs TARGET FILE URL EXPECTED_MD5)
  set(multiValueArgs PATCH_FILES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT ARG_FILE)
    message(FATAL_ERROR "morphizen_add_remote_target: FILE not specified")
  endif()
  if(NOT ARG_URL)
    message(FATAL_ERROR "morphizen_add_remote_target: URL not specified")
  endif()
  if(ARG_PATCH_FILES)
    find_package(Patch REQUIRED)
  endif()
  add_custom_command (
    OUTPUT ${ARG_FILE}
    COMMAND ${CMAKE_COMMAND}
    -DURL=${ARG_URL}
    -DFILE=${ARG_FILE}
    -DEXPECTED_MD5=${ARG_EXPECTED_MD5}
    -DPATCH_FILES=${ARG_PATCH_FILES}
    -DPatch_EXECUTABLE=${Patch_EXECUTABLE}
    -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/download.cmake
  )
  if(ARG_TARGET)
    add_custom_target(${ARG_TARGET} DEPENDS ${ARG_FILE})
  endif()
endfunction()

function(morphizen_add_python_target)
  set(options)
  set(oneValueArgs TARGET SCRIPT FOLDER)
  set(multiValueArgs ARGS DEPENDS OUTPUT)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT ARG_TARGET)
    message(FATAL_ERROR "morphizen_add_python_target: TARGET not specified")
  endif()
  if(NOT ARG_SCRIPT)
    message(FATAL_ERROR "morphizen_add_python_target: SCRIPT not specified")
  endif()
  add_custom_command (
    OUTPUT ${ARG_OUTPUT}
    COMMAND
      ${CMAKE_COMMAND} -E echo " -- Generating ${ARG_OUTPUT}"
    COMMAND
      ${CMAKE_COMMAND} -E echo " -- Running $<TARGET_FILE:Python3::Interpreter> ${ARG_SCRIPT} ${ARG_ARGS}"
    COMMAND
      ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../cmake/scripts"
      $<TARGET_FILE:Python3::Interpreter> -m pip install --user numpy onnx
    COMMAND
      ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../cmake/scripts"
      $<TARGET_FILE:Python3::Interpreter> ${ARG_SCRIPT} ${ARG_ARGS}
    DEPENDS ${ARG_SCRIPT} ${ARG_DEPENDS}
  )
  add_custom_target(${ARG_TARGET}
    DEPENDS ${ARG_OUTPUT}
    COMMENT "Generating ${ARG_OUTPUT} by runnning $<TARGET_FILE:Python3::Interpreter> ${ARG_SCRIPT} ${ARG_ARGS}"
  )
  if(ARG_FOLDER)
    set_target_properties(${ARG_TARGET} PROPERTIES FOLDER "${ARG_FOLDER}")
  endif()
endfunction(morphizen_add_python_target)
