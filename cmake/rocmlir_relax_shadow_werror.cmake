##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

function(_hip_ep_relax_shadow_werror dir)
  # rocmlirTriton's mlir/CMakeLists.txt sets LLVM_ENABLE_WERROR ON and adds
  # -Wshadow. Rock headers (GemmSize, TransformMapBuilder) take constructor
  # parameters that shadow members, which GCC treats as an error; clang-cl does
  # not enable that pair, so only the Linux rockCompiler build failed.
  #
  # A directory-scoped add_compile_options() cannot express this: at include
  # time the subdirectories do not exist yet, and after them it would not
  # attach to targets that are already defined. So walk the target tree and
  # append the flag per target instead.
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
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=shadow>)
  endforeach()
  get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(_sd IN LISTS _subdirs)
    _hip_ep_relax_shadow_werror("${_sd}")
  endforeach()
endfunction()

# build.py injects this file through CMAKE_PROJECT_TOP_LEVEL_INCLUDES, which
# runs inside the first project() call, so defer the walk to the end of the
# top-level CMakeLists once every target exists.
cmake_language(
  DEFER
  DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL _hip_ep_relax_shadow_werror "${CMAKE_SOURCE_DIR}")
