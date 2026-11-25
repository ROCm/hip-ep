##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# find_package(hipep_backend): libhipep_backend.so and hipep_backend.h as
# hipep_backend::hipep_backend. Installed in <prefix>/lib/cmake/hipep_backend.

get_filename_component(_hipep_backend_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT TARGET hipep_backend::hipep_backend)
  add_library(hipep_backend::hipep_backend SHARED IMPORTED)
  set_target_properties(hipep_backend::hipep_backend PROPERTIES
    IMPORTED_LOCATION "${_hipep_backend_prefix}/lib/libhipep_backend.so"
    INTERFACE_INCLUDE_DIRECTORIES "${_hipep_backend_prefix}/include")
endif()
set(hipep_backend_FOUND TRUE)
