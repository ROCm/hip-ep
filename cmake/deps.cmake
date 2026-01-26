##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(FetchContent)

# Include LLVM/MLIR configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/llvm.cmake)
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
    morphizen-hipdnn=morphizen-hipdnn)
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
  # for CI checker, MorphiZen is checkout by default in the parent directory
  "${CMAKE_SOURCE_DIR}/.."
  NO_DEFAULT_PATH)
if(MORPHIZEN_CMAKE_LIST_TXT_IN_LOCAL_WORKING_DIR)
  message(STATUS "found morphizen source in local working directory")
  message(STATUS "MorphiZen SOURCE_DIR ${MORPHIZEN_CMAKE_LIST_TXT_IN_LOCAL_WORKING_DIR}")
  FetchContent_Declare(
   morphizen
   SOURCE_DIR ${MORPHIZEN_CMAKE_LIST_TXT_IN_LOCAL_WORKING_DIR})
else()
  FetchContent_Declare(
  morphizen
  GIT_REPOSITORY https://github.com/ROCm/MorphiZen.git
  GIT_TAG main
  GIT_SUBMODULES "3rd-party/hash-library"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
endif()
set(morphizen_ENABLE_UNIT_TEST ON CACHE BOOL "enable vaip unit test or not")
if(morphizen_ENABLE_UNIT_TEST)
  include(CTest)
  enable_testing()
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.txt" "${VERSION_INFO}")
set(VAIP_VERSEION_INFO_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.txt")
set(VAIP_JSON_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/etc/vaip_config.json")

# Force static linking for glog to avoid runtime library conflicts in Debug mode
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(GLOG_BUILD_SHARED OFF CACHE BOOL "Build glog shared library" FORCE)

# Enable ORT bridge and MLIR backend for morphizen
set(morphizen_ENABLE_ORT_BRIDGE ON CACHE BOOL "Enable ORT bridge" FORCE)
set(morphizen_ENABLE_MLIR_BACKEND ON CACHE BOOL "Enable MLIR backend" FORCE)
set(morphizen_ENABLE_ONNX_BACKEND OFF CACHE BOOL "Enable ONNX backend" FORCE)
set(morphizen_ENABLE_ONNX_SCHEMA_SUPPORT OFF CACHE BOOL "Enable ONNX schema support" FORCE)
set(morphizen_ENABLE_RYZENAI_BIN_METADATA OFF CACHE BOOL "Disable ryzenai_bin_metadata submodule for version resource generation" FORCE)
set(morphizen_ENABLE_BOOST OFF CACHE BOOL "Disable Boost dependency" FORCE)

FetchContent_MakeAvailable(morphizen)
