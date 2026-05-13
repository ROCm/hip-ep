##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(FetchContent)

# Function to get git version info for a component
function(morphizen_add_version_info)
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

# Collect version info for onnx-hipdnn-ep
set(VERSION_LIST
    onnx-hipdnn-ep=onnx-hipdnn-ep)
set(VERSION_INFO "")
foreach(COMP_PAIR IN LISTS VERSION_LIST)
  string(FIND "${COMP_PAIR}" "=" pos)
  string(SUBSTRING "${COMP_PAIR}" 0 ${pos} COMP)
  math(EXPR COMP_BEG "${pos} + 1")
  string(SUBSTRING "${COMP_PAIR}" ${COMP_BEG} -1 DIR)
  set(BUILD_DIR "${CMAKE_SOURCE_DIR}/../${DIR}/")

  string(TOLOWER COMP ${COMP})
  set(FETCH_SRC_DIR "${${COMP}_SOURCE_DIR}")
  if(EXISTS "${FETCH_SRC_DIR}" AND IS_DIRECTORY "${FETCH_SRC_DIR}")
    morphizen_add_version_info(COMPONENT ${COMP} DIR ${FETCH_SRC_DIR})
  elseif(EXISTS ${BUILD_DIR} AND IS_DIRECTORY ${BUILD_DIR})
    morphizen_add_version_info(COMPONENT ${COMP} DIR ${BUILD_DIR})
  endif()
  if (NOT DEFINED COMP_GIT_COMMIT OR NOT DEFINED COMP_VERSION)
    set(COMP_GIT_COMMIT "N/A")
    set(COMP_VERSION "N/A")
  endif()
  string(APPEND VERSION_INFO "${COMP};${COMP_GIT_COMMIT};${COMP_VERSION}\n")
  unset(COMP_GIT_COMMIT)
  unset(COMP_VERSION)
endforeach()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.txt" "${VERSION_INFO}")
set(MORPHIZEN_VERSION_INFO_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.txt")

## Use morphizen from git submodule
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/3rd-party/morphizen/CMakeLists.txt")
  message(FATAL_ERROR "MorphiZen submodule not found. Run: git submodule update --init --recursive")
endif()
message(STATUS "Using MorphiZen from git submodule: 3rd-party/morphizen")

# Force static linking for glog to avoid runtime library conflicts
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(GLOG_BUILD_SHARED OFF CACHE BOOL "Build glog shared library" FORCE)

# Enable ORT bridge and MLIR backend (same as onnx-hipdnn-ep)
set(morphizen_ENABLE_RYZENAI_BIN_METADATA OFF CACHE BOOL "Disable ryzenai_bin_metadata" FORCE)
set(morphizen_OUTPUT_NAME "onnxruntime_morphizen_ep" CACHE STRING "Set output name" FORCE)
set(MORPHIZEN_VERSEION_INFO_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.txt")
set(MORPHIZEN_JSON_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/etc/morphizen_config.json")

# Cross-wire BUILD_MOCK_RUNTIME (this project) <-> morphizen's HIP GPU
# allocator option so the same build invocation does the right thing on
# both sides:
#
#   * Real build (BUILD_MOCK_RUNTIME=OFF): we want morphizen's HIP allocator,
#     which means find_package(hip) needs HIP_PLATFORM seeded *before*
#     add_subdirectory(3rd-party/morphizen) below; otherwise TheRock's
#     hip-config.cmake errors out with "Unexpected HIP_PLATFORM:".
#     (3rd-party/custom_kernels/cmake/hip_utils.cmake seeds it too, but
#     that subdir is added later in the top-level CMakeLists.txt.)
#     Morphizen's option default is already ON, so we don't have to FORCE
#     it -- but we don't actively turn it off either.
#
#   * Mock build (BUILD_MOCK_RUNTIME=ON, the project default): the toolchain
#     by definition has no ROCm SDK, so find_package(hip REQUIRED) inside
#     morphizen's ort-bridge would fail at configure time. Force morphizen's
#     allocator OFF so the EP DLL still compiles in mock mode (it just won't
#     register a HIP-backed allocator -- which is fine, mock can't run on
#     GPU anyway). Morphizen's own configure-time WARNING for ORT_BRIDGE +
#     ALLOCATOR=OFF is the expected, advertised-behavior signal here.
#
# This deps.cmake is included from the top-level CMakeLists.txt before its
# own `option(BUILD_MOCK_RUNTIME ...)` declaration, so we re-declare it
# here (option() keeps the value if already set on the command line via
# -DBUILD_MOCK_RUNTIME=...). Keep the default in sync with the top-level.
option(BUILD_MOCK_RUNTIME "Build mock runtime (no GPU required)" ON)

if(BUILD_MOCK_RUNTIME)
  set(morphizen_ENABLE_HIP_GPU_ALLOCATOR OFF CACHE BOOL "disabled in mock builds (no ROCm SDK available)" FORCE)
else()
  if(NOT DEFINED HIP_PLATFORM)
    set(HIP_PLATFORM "amd" CACHE STRING "HIP platform (amd or nvidia)")
  endif()
endif()

# cpptrace for crash backtraces
set(_saved_bsl_cpptrace ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
if(WIN32)
  set(CPPTRACE_UNWIND_WITH_WINAPI ON CACHE BOOL "" FORCE)
  set(CPPTRACE_UNWIND_WITH_DBGHELP OFF CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(
  cpptrace
  GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
  GIT_TAG v0.8.3
)
FetchContent_MakeAvailable(cpptrace)
set(BUILD_SHARED_LIBS ${_saved_bsl_cpptrace})

# Add morphizen subdirectory (after all options are set)
#
# morphizen's compile_options.linux.cmake defaults MORPHIZEN_COMPILER_OPTIONS
# to "-Wall -Werror -Wconversion -pedantic -Wextra -fPIC" via CACHE STRING (no
# FORCE). On Linux this propagates into the morphizen-* targets, including the
# ones that include protobuf-generated *.pb.h headers. GCC 13+ emits
# -Wconversion on the boilerplate that protobuf >=22 generates (e.g.
# `return _internal_foo().size();` returns size_t but the accessor signature
# is `int`), and the morphizen `-Werror` then upgrades those to hard errors.
#
# Override the cache value with `FORCE` before add_subdirectory(morphizen) so
# the new value sticks across reconfigures: keep -Wconversion as a warning
# (still useful as a signal during code review) but stop -Werror from
# promoting it to a build failure. Other diagnostics (sign-compare, etc.)
# remain -Werror. Windows / MSVC users are unaffected because the equivalent
# /WX-only knob lives in compile_options.msvc.cmake.
if(UNIX AND NOT APPLE)
    set(MORPHIZEN_COMPILER_OPTIONS
        -Wall -Werror -Wconversion -Wno-error=conversion -pedantic
        -Wextra -fPIC
        CACHE STRING "Compiler options for Morphizen" FORCE)
endif()

add_subdirectory(3rd-party/morphizen)
