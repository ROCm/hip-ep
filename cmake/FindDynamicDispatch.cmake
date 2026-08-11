##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
#
# FindDynamicDispatch.cmake
#
# Locates AMD DynamicDispatch library (Vitis AI / XRT runtime)
#
# Sets:
#   DynamicDispatch_FOUND - TRUE if DynamicDispatch library and headers found
#   DynamicDispatch_INCLUDE_DIRS - Include directories for DynamicDispatch
#   DynamicDispatch_LIBRARIES - Library path for DynamicDispatch
#
# Defines imported target:
#   DynamicDispatch::DynamicDispatch
#
# Environment variables checked:
#   DYNAMICDISPATCH_ROOT - Root directory of DynamicDispatch installation
#   VAI_RT_ROOT - Alternative: Vitis AI Runtime root directory
#

# Check environment variables for installation path
if(DEFINED ENV{DYNAMICDISPATCH_ROOT})
  set(_DD_ROOT "$ENV{DYNAMICDISPATCH_ROOT}")
elseif(DEFINED ENV{VAI_RT_ROOT})
  set(_DD_ROOT "$ENV{VAI_RT_ROOT}")
else()
  # Default search paths
  set(_DD_ROOT "")
endif()

# Search for include directory
find_path(DynamicDispatch_INCLUDE_DIR
  NAMES ops/op_interface.hpp
  PATHS
    ${_DD_ROOT}
    "c:/Vai-rt-0/Vai-rt-build/Install"
  PATH_SUFFIXES
    Include/Ryzenai/Dynamic_dispatch
    include/Ryzenai/Dynamic_dispatch
    include
  DOC "DynamicDispatch include directory"
)

# Search for library
find_library(DynamicDispatch_LIBRARY
  NAMES dynamic_dispatch dd_helper
  PATHS
    ${_DD_ROOT}
    "c:/Vai-rt-0/Vai-rt-build/Install"
  PATH_SUFFIXES
    Lib
    lib
    lib64
  DOC "DynamicDispatch library"
)

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DynamicDispatch
  REQUIRED_VARS
    DynamicDispatch_INCLUDE_DIR
    DynamicDispatch_LIBRARY
  FAIL_MESSAGE
    "DynamicDispatch not found. Set DYNAMICDISPATCH_ROOT or VAI_RT_ROOT environment variable."
)

if(DynamicDispatch_FOUND)
  set(DynamicDispatch_INCLUDE_DIRS ${DynamicDispatch_INCLUDE_DIR})
  set(DynamicDispatch_LIBRARIES ${DynamicDispatch_LIBRARY})

  # Create imported target
  if(NOT TARGET DynamicDispatch::DynamicDispatch)
    add_library(DynamicDispatch::DynamicDispatch UNKNOWN IMPORTED)
    set_target_properties(DynamicDispatch::DynamicDispatch PROPERTIES
      IMPORTED_LOCATION "${DynamicDispatch_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${DynamicDispatch_INCLUDE_DIR}"
    )
  endif()

  message(STATUS "Found DynamicDispatch:")
  message(STATUS "  Include: ${DynamicDispatch_INCLUDE_DIR}")
  message(STATUS "  Library: ${DynamicDispatch_LIBRARY}")
endif()

mark_as_advanced(
  DynamicDispatch_INCLUDE_DIR
  DynamicDispatch_LIBRARY
)
