##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(FetchContent)

# Include LLVM/MLIR configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/llvm.cmake)

# Read and parse deps.txt for dependency version management
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/deps.txt MORPHIZEN_DEPS_LIST)
file(READ "${CMAKE_CURRENT_LIST_DIR}/dep.h.inc.in" MORPHIZEN_DEP_H_INC_IN)
set(MORPHIZEN_DEP_H_INC "")

foreach(MORPHIZEN_DEP IN LISTS MORPHIZEN_DEPS_LIST)
  # Lines start with "#" are comments
  if(NOT MORPHIZEN_DEP MATCHES "^#")
    message(STATUS "MORPHIZEN_DEP = ${MORPHIZEN_DEP}")
    # The first column is name
    list(POP_FRONT MORPHIZEN_DEP MORPHIZEN_DEP_NAME)
    # The second column is URL
    list(POP_FRONT MORPHIZEN_DEP MORPHIZEN_DEP_URL)
    set(DEP_URL_${MORPHIZEN_DEP_NAME} ${MORPHIZEN_DEP_URL})
    # The third column is SHA1 hash value or Git tag
    set(DEP_TAG_${MORPHIZEN_DEP_NAME} ${MORPHIZEN_DEP})
    
    # Determine if this is a Git repository
    if(MORPHIZEN_DEP_URL MATCHES "\\.git$")
      set(DEP_IS_GIT_${MORPHIZEN_DEP_NAME} TRUE)
      message(STATUS "  -> Git repository: ${MORPHIZEN_DEP_URL} @ ${MORPHIZEN_DEP}")
    else()
      set(DEP_IS_GIT_${MORPHIZEN_DEP_NAME} FALSE)
      set(DEP_SHA1_${MORPHIZEN_DEP_NAME} ${MORPHIZEN_DEP})
      message(STATUS "  -> Archive: ${MORPHIZEN_DEP_URL} (SHA1: ${MORPHIZEN_DEP})")
    endif()
    
    string(CONFIGURE "${MORPHIZEN_DEP_H_INC_IN}" _tmp @ONLY)
    string(APPEND MORPHIZEN_DEP_H_INC "${_tmp}")
  endif()
endforeach()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/morphizen_deps.inc.h" "${MORPHIZEN_DEP_H_INC}")

function(vaip_add_version_info)
  set(options)
  set(oneValueArgs COMPONENT DIR)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})
  if(EXISTS "${ARG_DIR}/.git" AND IS_DIRECTORY "${ARG_DIR}/.git")
    execute_process(
      COMMAND "git" rev-parse HEAD
      WORKING_DIRECTORY ${ARG_DIR}
      OUTPUT_VARIABLE TMP_GIT_COMMIT
      ERROR_VARIABLE error
      RESULT_VARIABLE resultVar
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
      COMMAND "git" describe --tags --abbrev=1 HEAD
      WORKING_DIRECTORY ${ARG_DIR}
      OUTPUT_VARIABLE TMP_VERSION
      ERROR_VARIABLE error
      RESULT_VARIABLE resultVar
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  set(COMP_GIT_COMMIT ${TMP_GIT_COMMIT} PARENT_SCOPE)
  set(COMP_VERSION ${TMP_VERSION} PARENT_SCOPE)
  message(STATUS "FindPackage Version info: ${ARG_COMPONENT}=${TMP_GIT_COMMIT} ${TMP_VERSION}")
endfunction()

set(VERSION_LIST
    onnx-hipdnn-ep=onnx-hipdnn-ep)
set(VERSION_INFO "")
foreach(COMP_PAIR IN LISTS VERSION_LIST)
  string(FIND "${COMP_PAIR}" "=" pos)
  string(SUBSTRING "${COMP_PAIR}" 0 ${pos} COMP)
  math(EXPR COMP_BEG "${pos} + 1")
  string(SUBSTRING "${COMP_PAIR}" ${COMP_BEG} -1 DIR)
  set(BUILD_DIR "${CMAKE_SOURCE_DIR}/../${DIR}/") # already built source dir

  string(TOLOWER COMP ${COMP})
  set(FETCH_SRC_DIR "${${COMP}_SOURCE_DIR}")
  if(EXISTS "${FETCH_SRC_DIR}" AND IS_DIRECTORY "${FETCH_SRC_DIR}")
    vaip_add_version_info(COMPONENT ${COMP} DIR ${FETCH_SRC_DIR})
  elseif(EXISTS ${BUILD_DIR} AND IS_DIRECTORY ${BUILD_DIR})
    vaip_add_version_info(COMPONENT ${COMP} DIR ${BUILD_DIR})
  endif()
  if (NOT DEFINED COMP_GIT_COMMIT OR NOT DEFINED COMP_VERSION)
    set(COMP_GIT_COMMIT "N/A")
    set(COMP_VERSION "N/A")
  endif()
  string(APPEND VERSION_INFO "${COMP};${COMP_GIT_COMMIT};${COMP_VERSION}\n")
  unset(COMP_GIT_COMMIT)
  unset(COMP_VERSION)
endforeach()
## if morphizen source is checked out from git in the parent directory, we use the working directory from there.
find_path(MORPHIZEN_CMAKE_LIST_TXT_IN_LOCAL_WORKING_DIR
  NAMES CMakeLists.txt
  PATHS "${CMAKE_SOURCE_DIR}/../morphizen"
  "${CMAKE_SOURCE_DIR}/../MorphiZen"
  # for CI checkder, MophiZen is checkout by default in the parent directory
  "${CMAKE_SOURCE_DIR}/.."
  NO_DEFAULT_PATH)
if(MORPHIZEN_CMAKE_LIST_TXT_IN_LOCAL_WORKING_DIR)
  message(STATUS "found morphizen source in local working directory")
  message(STATUS "MorphiZen SOURCE_DIR ${MORPHIZEN_CMAKE_LIST_TXT_IN_LOCAL_WORKING_DIR}")
  FetchContent_Declare(
   morphizen
   SOURCE_DIR ${MORPHIZEN_CMAKE_LIST_TXT_IN_LOCAL_WORKING_DIR})
else()
  message(STATUS "Fetching morphizen from deps.txt: ${DEP_URL_morphizen} @ ${DEP_TAG_morphizen}")
  FetchContent_Declare(
  morphizen
  GIT_REPOSITORY ${DEP_URL_morphizen}
  GIT_TAG ${DEP_TAG_morphizen}
  GIT_SUBMODULES "3rd-party/hash-library"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
endif()
set(morphizen_ENABLE_UNIT_TEST ON CACHE BOOL "enable vaip unit test or not")
if(morphizen_ENABLE_UNIT_TEST)
  include(CTest)
  enable_testing()
endif()

# Ensure ONNX uses dynamic runtime to match CMAKE_MSVC_RUNTIME_LIBRARY setting
# This fixes the LNK2038 runtime library mismatch errors
set(ONNX_USE_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Use static MSVC runtime" FORCE)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.txt" "${VERSION_INFO}")
set(VAIP_VERSEION_INFO_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.txt")
set(MORPHIZEN_JSON_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/etc/morphizen_config.json")

# Enable ORT bridge and MLIR backend for morphizen
set(morphizen_ENABLE_ORT_BRIDGE ON CACHE BOOL "Enable ORT bridge" FORCE)
set(morphizen_ENABLE_MLIR_BACKEND ON CACHE BOOL "Enable MLIR backend" FORCE)
set(morphizen_ENABLE_ONNX_BACKEND OFF CACHE BOOL "Enable ONNX backend" FORCE)
set(morphizen_ENABLE_ONNX_SCHEMA_SUPPORT OFF CACHE BOOL "Enable ONNX schema support" FORCE)
set(morphizen_ENABLE_RYZENAI_BIN_METADATA OFF CACHE BOOL "Disable ryzenai_bin_metadata submodule for version resource generation" FORCE)
set(morphizen_ENABLE_BOOST OFF CACHE BOOL "Disable Boost dependency" FORCE)
set(morphizen_OUTPUT_NAME "onnxruntime_morphizen_ep" CACHE STRING "Set morphizen output name of Morphizen" FORCE)

FetchContent_MakeAvailable(morphizen)
