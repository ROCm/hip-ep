##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(FetchContent)

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

message(STATUS "Configuring LLVM/MLIR for onnx-hipdnn-ep")

# LLVM configuration options
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "LLVM projects to build")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "LLVM targets to build")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable LLVM assertions")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "Enable RTTI in LLVM")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "Enable libedit in LLVM")
set(LLVM_BUILD_TOOLS ON CACHE BOOL "Build LLVM tools")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "Install LLVM utilities")
set(LLVM_INCLUDE_TESTS ON CACHE BOOL "Build LLVM tests")
set(LLVM_DISABLE_ASSEMBLY_FILES OFF CACHE BOOL "disable assembly")
set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "Use static zlib")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "Enable zstd compression")

# LLVM commit hash and URL are now managed in deps.txt
# These variables are set by deps.cmake parsing logic
# DEP_URL_llvm and DEP_TAG_llvm are defined from deps.txt

# Try to find pre-installed LLVM/MLIR first
find_package(LLVM QUIET CONFIG)
find_package(MLIR QUIET CONFIG)

if(LLVM_FOUND AND MLIR_FOUND)
  message(STATUS "Found pre-installed LLVM and MLIR")
  message(STATUS "LLVM_DIR: ${LLVM_DIR}")
  message(STATUS "MLIR_DIR: ${MLIR_DIR}")
  set(MORPHIZEN_LLVM_PREINSTALLED ON CACHE BOOL "Using pre-installed LLVM" FORCE)
else()
  message(STATUS "LLVM/MLIR not found in CMAKE_PREFIX_PATH, will use FetchContent and build inline")
  set(MORPHIZEN_LLVM_PREINSTALLED OFF CACHE BOOL "Using FetchContent LLVM" FORCE)
  
  # Try to find LLVM in local directories first
  find_path(LOCAL_LLVM
    NAMES CMakeLists.txt
    PATHS
    "${CMAKE_SOURCE_DIR}/../llvm/llvm"
    "${CMAKE_SOURCE_DIR}/llvm/llvm"
    "${CMAKE_SOURCE_DIR}/3rd-party/llvm/llvm"
    "${CMAKE_SOURCE_DIR}/../llvm-project/llvm"
    "${CMAKE_SOURCE_DIR}/llvm-project/llvm"
    "${CMAKE_SOURCE_DIR}/3rd-party/llvm-project/llvm"
    NO_DEFAULT_PATH)
  
  if(LOCAL_LLVM)
    message(STATUS "Found LLVM source in local working directory")
    message(STATUS "LLVM SOURCE_DIR: ${LOCAL_LLVM}")
    FetchContent_Declare(
      llvm-project
      SOURCE_DIR ${LOCAL_LLVM}/..
      SOURCE_SUBDIR llvm)
  else()
    message(STATUS "Cannot find LLVM in local directories")
    message(STATUS "Fetching LLVM source from deps.txt: ${DEP_URL_llvm} @ ${DEP_TAG_llvm}")
    message(STATUS "WARNING: This will download and build LLVM, which may take a long time")
    FetchContent_Declare(
      llvm-project
      GIT_REPOSITORY ${DEP_URL_llvm}
      GIT_TAG ${DEP_TAG_llvm}
      GIT_SUBMODULES_RECURSE
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      EXCLUDE_FROM_ALL
      SOURCE_SUBDIR llvm
    )
  endif()
  
  # Make LLVM available - this adds LLVM as a subdirectory
  FetchContent_MakeAvailable(llvm-project)
  
  # Set include directories for downstream targets
  set(LLVM_INCLUDE_DIRS 
    "${llvm-project_SOURCE_DIR}/llvm/include"
    "${llvm-project_BINARY_DIR}/include"
    CACHE PATH "LLVM include directories" FORCE)
  set(MLIR_INCLUDE_DIRS 
    "${llvm-project_SOURCE_DIR}/mlir/include"
    "${llvm-project_BINARY_DIR}/tools/mlir/include"
    CACHE PATH "MLIR include directories" FORCE)
  
  # Make include directories globally available for all targets
  # This is necessary because subdirectory builds don't automatically propagate
  # MLIR source includes to targets that use find_package(MLIR)
  include_directories(SYSTEM
    "${llvm-project_SOURCE_DIR}/llvm/include"
    "${llvm-project_BINARY_DIR}/include"
    "${llvm-project_SOURCE_DIR}/mlir/include"
    "${llvm-project_BINARY_DIR}/tools/mlir/include")
  
  # Note: In a subdirectory build, MLIR config files are in tools/mlir/cmake/modules/CMakeFiles/
  set(LLVM_DIR "${llvm-project_BINARY_DIR}/lib/cmake/llvm" CACHE PATH "" FORCE)
  set(MLIR_DIR "${llvm-project_BINARY_DIR}/tools/mlir/cmake/modules/CMakeFiles" CACHE PATH "" FORCE)
  
  message(STATUS "LLVM source dir: ${llvm-project_SOURCE_DIR}")
  message(STATUS "LLVM binary dir: ${llvm-project_BINARY_DIR}")
  message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
  message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
  message(STATUS "LLVM_DIR: ${LLVM_DIR}")
  message(STATUS "MLIR_DIR: ${MLIR_DIR}")
endif()

message(STATUS "LLVM/MLIR configuration complete")

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
  message(STATUS "Fetching morphizen from deps.txt: ${DEP_URL_morphizen} @ ${DEP_TAG_morphizen}")
  FetchContent_Declare(
  morphizen
  GIT_REPOSITORY ${DEP_URL_morphizen}
  GIT_TAG ${DEP_TAG_morphizen}
  GIT_SUBMODULES "3rd-party/hash-library"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
endif()
set(morphizen_ENABLE_UNIT_TEST ON CACHE BOOL "enable morphizen unit test or not")
if(morphizen_ENABLE_UNIT_TEST)
  include(CTest)
  enable_testing()
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.txt" "${VERSION_INFO}")
set(MORPHIZEN_VERSION_INFO_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.txt")

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
set(morphizen_OUTPUT_NAME "onnxruntime_morphizen_ep" CACHE STRING "Set output name of Morphizen Library" FORCE)
set(MORPHIZEN_JSON_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/etc/morphizen_config.json" CACHE FILEPATH "Path to morphizen config file" FORCE)

# Silence MLIR std::complex<APFloat> deprecation warning (MSVC)
if(MSVC)
  add_compile_definitions(_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING)
endif()

# Override the output name to use onnxruntime_morphizen_ep instead of onnxruntime_vitisai_ep
set(morphizen_OUTPUT_NAME "onnxruntime_morphizen_ep" CACHE STRING "Output name of MorphiZen library" FORCE)

FetchContent_MakeAvailable(morphizen)
