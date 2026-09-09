##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

if(NOT DEFINED LLVM_SOURCE_DIR OR NOT EXISTS "${LLVM_SOURCE_DIR}/llvm/CMakeLists.txt")
  message(FATAL_ERROR "LLVM_SOURCE_DIR must name an llvm-project source tree")
endif()

# These are the patch records accompanying the LLVM subtree in the pinned
# rocmlirTriton revision. Keep the order stable: some scheduler fixes are
# follow-ups to earlier patches.
set(_rocmlir_patch_names
  patch167347.patch
  patch211809.patch
  patch-gfx1170-ocp-fp8.patch
  patch-rocm-runtime-dllexport.patch
  patch205651.patch
  patch200945.patch
  patch202743.patch
  patch204648.patch
  patch205637.patch
  patch208045.patch
  patch208280.patch
  patch209704.patch
  patch211884.patch
  patch201494.patch
  patch203770.patch)
set(_rocmlir_patch_hashes
  f3c02dbfb122b09e24fd14f1b32a26e29c1ae4408976ff2e339421383b8be01a
  cf7a1dbb27b2dfbfe385e0f4b1bf862348c09cc645d767e7b47010b4bc95a0cd
  49133a9487d67499666a9536ee243b8a043bcee71120fe9e98677856d518a4b0
  7317fd4e238e81c34af6c5e622de3ece6caed38e75f4448caab44854b293d9c4
  765744c0ae0f7e3171a4edaf804ac1e5f09d7fb588a09d8629c276f8ab06f4c9
  5938db2651c791015beb08066e498520b0e68054e83bf3b580a056c912356c2d
  ab84337111d8cdf83611d139452179bb632b80ed7d899abee5ea855b1962dfd6
  389d97b4486f1f874ff4350502ea650eb133f86f54adf89f748eece4aad1199a
  bdf0306d9b076c439e15bd34d0847b64cc81ce4d1a113390c9c58d255e799660
  308c3b5750baf209f119f4f4db99583fbe8188ebf791a8d413965c4a9f6ea3b4
  1e9af219c712a7ef9b92bfa99aa5afef2bdffbf2a315a604ea4567a9601c4173
  8cd92ca4a75601810d6cfd1f1b13615fb928eb625512fef64386538e3c9cf0c7
  abc8d59e6fa7b8b0936c3a2658dc4b16d5dda4ded9c11f85ceeca4e2fbcf3084
  120ce89cde8259ff479d11199fc5bd7cfbfd5395802b667e1d127b6519eedc2d
  c3c61e777287144a9d62960c98b4e103464d49cfa82b0a639d05db07e96ac990)

list(LENGTH _rocmlir_patch_names _name_count)
list(LENGTH _rocmlir_patch_hashes _hash_count)
if(NOT _name_count EQUAL _hash_count)
  message(FATAL_ERROR "rocmlirTriton LLVM patch name/hash lists are inconsistent")
endif()

set(_patch_cache "${CMAKE_CURRENT_BINARY_DIR}/rocmlir-llvm-patches")
file(MAKE_DIRECTORY "${_patch_cache}")
math(EXPR _last_patch "${_name_count} - 1")

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

foreach(_index RANGE ${_last_patch})
  list(GET _rocmlir_patch_names ${_index} _name)
  list(GET _rocmlir_patch_hashes ${_index} _sha256)

  if(DEFINED ROCMLIRTRITON_SOURCE_DIR AND
     EXISTS "${ROCMLIRTRITON_SOURCE_DIR}/llvm-patches/${_name}")
    set(_patch "${ROCMLIRTRITON_SOURCE_DIR}/llvm-patches/${_name}")
    file(SHA256 "${_patch}" _actual_sha256)
    if(NOT _actual_sha256 STREQUAL _sha256)
      message(FATAL_ERROR
        "Unexpected checksum for ${_patch}: ${_actual_sha256}, expected ${_sha256}")
    endif()
  else()
    set(_patch "${_patch_cache}/${_name}")
    set(_url
      "https://raw.githubusercontent.com/ROCm/rocmlirTriton/28a5a7a40cfd6aceab0047c7fc92cb5cf90b6540/llvm-patches/${_name}")
    file(DOWNLOAD "${_url}" "${_patch}"
      EXPECTED_HASH "SHA256=${_sha256}"
      STATUS _download_status)
    list(GET _download_status 0 _download_code)
    if(NOT _download_code EQUAL 0)
      message(FATAL_ERROR "Failed to download ${_url}: ${_download_status}")
    endif()
  endif()

  execute_process(
    COMMAND git apply ${_git_apply_directory_arg}
            --check --whitespace=nowarn "${_patch}"
    WORKING_DIRECTORY "${_git_root}"
    RESULT_VARIABLE _check_result
    OUTPUT_QUIET ERROR_QUIET)
  if(_check_result EQUAL 0)
    execute_process(
      COMMAND git apply ${_git_apply_directory_arg}
              --whitespace=nowarn "${_patch}"
      WORKING_DIRECTORY "${_git_root}"
      RESULT_VARIABLE _apply_result)
    if(NOT _apply_result EQUAL 0)
      message(FATAL_ERROR "Failed to apply rocmlirTriton LLVM patch ${_name}")
    endif()
    message(STATUS "Applied rocmlirTriton LLVM patch ${_name}")
  else()
    execute_process(
      COMMAND git apply ${_git_apply_directory_arg}
              --reverse --check --whitespace=nowarn "${_patch}"
      WORKING_DIRECTORY "${_git_root}"
      RESULT_VARIABLE _reverse_result
      OUTPUT_QUIET ERROR_QUIET)
    if(_reverse_result EQUAL 0)
      message(STATUS "rocmlirTriton LLVM patch already applied: ${_name}")
    else()
      message(FATAL_ERROR
        "rocmlirTriton LLVM patch does not apply cleanly: ${_name}")
    endif()
  endif()
endforeach()
