##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

if(NOT RYZENAICORELIB_ROOT AND DEFINED ENV{RYZENAICORELIB_ROOT})
  set(RYZENAICORELIB_ROOT "$ENV{RYZENAICORELIB_ROOT}")
endif()

# Finds only the RyzenAI CoreLib header: the runtime loads the library on first
# use. RYZENAICORELIB_ROOT is a corelib install tree or repository root.
# Sets RyzenAICorelib_FOUND, RyzenAICorelib_INCLUDE_DIR, RyzenAICorelib_VERSION.
find_path(RyzenAICorelib_INCLUDE_DIR
  NAMES ryzenai/corelib.h
  HINTS ${RYZENAICORELIB_ROOT}
  PATH_SUFFIXES include install/include
  DOC "RyzenAI CoreLib include directory"
)

if(RyzenAICorelib_INCLUDE_DIR)
  file(STRINGS "${RyzenAICorelib_INCLUDE_DIR}/ryzenai/corelib.h" _corelib_ver
       REGEX "^#define RYZENAI_CORELIB_VERSION_(MAJOR|MINOR|PATCH) [0-9]+")
  foreach(_part MAJOR MINOR PATCH)
    string(REGEX REPLACE ".*VERSION_${_part} ([0-9]+).*" "\\1" _v_${_part}
           "${_corelib_ver}")
  endforeach()
  set(RyzenAICorelib_VERSION "${_v_MAJOR}.${_v_MINOR}.${_v_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RyzenAICorelib
  REQUIRED_VARS RyzenAICorelib_INCLUDE_DIR
  VERSION_VAR RyzenAICorelib_VERSION
)

mark_as_advanced(RyzenAICorelib_INCLUDE_DIR)
