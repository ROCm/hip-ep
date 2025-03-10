function(vaip_add_remote_target)
  set(options)
  set(oneValueArgs TARGET FILE URL EXPECTED_MD5)
  set(multiValueArgs PATCH_FILES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT ARG_FILE)
    message(FATAL_ERROR "vaip_add_remote_target: FILE not specified")
  endif()
  if(NOT ARG_URL)
    message(FATAL_ERROR "vaip_add_remote_target: URL not specified")
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
