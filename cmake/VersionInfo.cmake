##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

if(MSVC)
  enable_language(RC)
endif()

set(HIPEP_VERSION_RC_TEMPLATE "${CMAKE_CURRENT_LIST_DIR}/version.rc.in")

function(hipdnn_ep_apply_binary_compliance target)
  if(NOT MSVC)
    return()
  endif()

  cmake_parse_arguments(ARG "" "DESCRIPTION" "" ${ARGN})

  target_link_options(${target} PRIVATE /guard:cf /DYNAMICBASE)

  get_target_property(_output_name ${target} OUTPUT_NAME)
  if(NOT _output_name)
    set(_output_name ${target})
  endif()

  set(HIPEP_RC_VERSION_CSV
    "${CMAKE_PROJECT_VERSION_MAJOR},${CMAKE_PROJECT_VERSION_MINOR},${CMAKE_PROJECT_VERSION_PATCH},0")
  set(HIPEP_RC_VERSION_DOTTED "${CMAKE_PROJECT_VERSION}.0")
  set(HIPEP_RC_INTERNAL_NAME "${_output_name}")
  set(HIPEP_RC_ORIGINAL_FILENAME "${_output_name}.dll")
  set(HIPEP_RC_FILE_DESCRIPTION "${ARG_DESCRIPTION}")
  string(TIMESTAMP HIPEP_RC_YEAR "%Y" UTC)

  set(_rc "${CMAKE_CURRENT_BINARY_DIR}/${target}_version.rc")
  configure_file("${HIPEP_VERSION_RC_TEMPLATE}" "${_rc}" @ONLY)
  target_sources(${target} PRIVATE "${_rc}")
endfunction()
