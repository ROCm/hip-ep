##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

function(_hip_ep_relax_werror dir)
  # rocmlirTriton's mlir/CMakeLists.txt sets LLVM_ENABLE_WERROR ON and adds
  # -Wshadow -Wnull-dereference -Wformat=2 -Wundef -Wmissing-declarations on top
  # of LLVM's -Wall/-Wextra. Upstream validates that with Clang, and clang-cl is
  # what hip-ep uses on Windows, so only the GCC (Linux) build trips over it:
  #
  #   Rock/utility/GemmSize.h: -Werror=shadow (ctor params shadow members)
  #   mlir/IR/UseDefLists.h:   -Werror=null-dereference
  #   llvm/ADT/Hashing.h:      -Werror=maybe-uninitialized
  #
  # The last two are GCC false positives inside upstream LLVM and MLIR headers.
  # We own none of this code, so demote every warning instead of enumerating
  # them: the warnings still print, they just no longer fail a dependency build.
  # Enumerating cost one ~1h CI cycle per newly surfaced warning.
  #
  # Appending per target is what makes this work. -Wno-error has to land after
  # the -Werror that HandleLLVMOptions puts in CMAKE_CXX_FLAGS, and a
  # directory-scoped add_compile_options() cannot do it: at include time the
  # subdirectories do not exist yet, and after them it would not attach to
  # targets that are already defined.
  get_property(_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_t IN LISTS _targets)
    if(NOT TARGET "${_t}")
      continue()
    endif()
    get_target_property(_type "${_t}" TYPE)
    if(_type STREQUAL "INTERFACE_LIBRARY" OR _type STREQUAL "UTILITY")
      continue()
    endif()
    target_compile_options("${_t}" PRIVATE
      "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wno-error>"
      "$<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Wno-error>")
    get_property(_n GLOBAL PROPERTY _HIP_EP_RELAXED_TARGETS)
    set_property(GLOBAL PROPERTY _HIP_EP_RELAXED_TARGETS "${_n};${_t}")
  endforeach()
  get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(_sd IN LISTS _subdirs)
    _hip_ep_relax_werror("${_sd}")
  endforeach()
endfunction()

function(_hip_ep_report_relaxed_werror)
  # Print the count so a CI log shows whether the walk actually reached the
  # tree, rather than leaving a silent no-op to be discovered by a failed build.
  get_property(_relaxed GLOBAL PROPERTY _HIP_EP_RELAXED_TARGETS)
  list(REMOVE_ITEM _relaxed "")
  list(LENGTH _relaxed _n)
  message(STATUS "hip-ep: relaxed -Werror on ${_n} rocmlirTriton targets")
endfunction()

# build.py injects this file through CMAKE_PROJECT_TOP_LEVEL_INCLUDES, which
# runs inside the first project() call, so defer the walk to the end of the
# top-level CMakeLists once every target exists.
cmake_language(
  DEFER
  DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL _hip_ep_relax_werror "${CMAKE_SOURCE_DIR}")
cmake_language(
  DEFER
  DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL _hip_ep_report_relaxed_werror)
