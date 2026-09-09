##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

if(NOT DEFINED LLVM_SOURCE_DIR OR NOT EXISTS "${LLVM_SOURCE_DIR}/llvm/CMakeLists.txt")
  message(FATAL_ERROR "LLVM_SOURCE_DIR must name an llvm-project source tree")
endif()

# rocmlirTriton's downstream LLVM changes, generated from its vendored
# external/llvm-project rather than replayed from its llvm-patches/ records.
# The header of the patch explains why the records cannot be replayed.
set(_rocmlir_patch "${CMAKE_CURRENT_LIST_DIR}/patches/rocmlir-llvm-source.patch")
if(NOT EXISTS "${_rocmlir_patch}")
  message(FATAL_ERROR "Missing rocmlirTriton LLVM patch: ${_rocmlir_patch}")
endif()

execute_process(
  COMMAND git rev-parse --show-toplevel
  WORKING_DIRECTORY "${LLVM_SOURCE_DIR}"
  OUTPUT_VARIABLE _git_root
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _git_root_result)
if(NOT _git_root_result EQUAL 0)
  message(FATAL_ERROR "${LLVM_SOURCE_DIR} is not in a Git checkout")
endif()
get_filename_component(_git_root "${_git_root}" REALPATH)
get_filename_component(_llvm_source_real "${LLVM_SOURCE_DIR}" REALPATH)
file(RELATIVE_PATH _llvm_source_relative "${_git_root}" "${_llvm_source_real}")
set(_git_apply_directory_arg)
if(NOT _llvm_source_relative STREQUAL ".")
  # A local rocmlirTriton checkout is one Git repository with LLVM under
  # external/llvm-project. git apply anchors paths at the repository root.
  set(_git_apply_directory_arg "--directory=${_llvm_source_relative}")
endif()

execute_process(
  COMMAND git apply ${_git_apply_directory_arg}
          --check --whitespace=nowarn "${_rocmlir_patch}"
  WORKING_DIRECTORY "${_git_root}"
  RESULT_VARIABLE _check_result
  OUTPUT_QUIET ERROR_QUIET)

if(_check_result EQUAL 0)
  execute_process(
    COMMAND git apply ${_git_apply_directory_arg}
            --whitespace=nowarn "${_rocmlir_patch}"
    WORKING_DIRECTORY "${_git_root}"
    RESULT_VARIABLE _apply_result)
  if(NOT _apply_result EQUAL 0)
    message(FATAL_ERROR "Failed to apply the rocmlirTriton LLVM patch")
  endif()
  message(STATUS "Applied the rocmlirTriton LLVM patch")
else()
  execute_process(
    COMMAND git apply ${_git_apply_directory_arg}
            --reverse --check --whitespace=nowarn "${_rocmlir_patch}"
    WORKING_DIRECTORY "${_git_root}"
    RESULT_VARIABLE _reverse_result
    OUTPUT_QUIET ERROR_QUIET)
  if(_reverse_result EQUAL 0)
    # A rocmlirTriton checkout already carries these changes as subtree commits.
    message(STATUS "rocmlirTriton LLVM patch already applied")
  else()
    message(FATAL_ERROR
      "rocmlirTriton LLVM patch does not apply to ${LLVM_SOURCE_DIR}; the tree "
      "is not at the pinned upstream base")
  endif()
endif()
