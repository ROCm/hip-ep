##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ggml, for the split frontend and libhipep_backend (HIPDNN_EP_BUILD_GGML),
# linked statically: it must be the ggml the llama.cpp that loads
# libhipep_backend was built with, from HIPDNN_EP_LLAMA_CPP_SOURCE_DIR or the
# llama.cpp revision below. Defines ggml-base, ggml-cpu and ggml, NlohmannJson
# (llama.cpp's vendored nlohmann/json) and HIPDNN_EP_GGML_SOURCE_DIR.

include(FetchContent)

# The AMD-Ecosystem llama.cpp revision that carries ggml-hipep (its ggml-hipep
# branch).
set(HIPDNN_EP_LLAMA_CPP_REPOSITORY "https://github.com/AMD-Ecosystem/llama.cpp"
  CACHE STRING "llama.cpp repository ggml is taken from")
set(HIPDNN_EP_LLAMA_CPP_REVISION "f600e283d2a47e7697bad68acc042d6c52937319"
  CACHE STRING "llama.cpp revision ggml is taken from")
set(HIPDNN_EP_LLAMA_CPP_SOURCE_DIR "" CACHE PATH
  "A llama.cpp source tree to take ggml from instead of fetching it")

if(HIPDNN_EP_LLAMA_CPP_SOURCE_DIR)
  set(_llama_cpp_source_dir "${HIPDNN_EP_LLAMA_CPP_SOURCE_DIR}")
else()
  FetchContent_Declare(hipdnn_ep_llama_cpp
    URL "${HIPDNN_EP_LLAMA_CPP_REPOSITORY}/archive/${HIPDNN_EP_LLAMA_CPP_REVISION}.tar.gz")
  FetchContent_GetProperties(hipdnn_ep_llama_cpp)
  if(NOT hipdnn_ep_llama_cpp_POPULATED)
    FetchContent_Populate(hipdnn_ep_llama_cpp)
  endif()
  set(_llama_cpp_source_dir "${hipdnn_ep_llama_cpp_SOURCE_DIR}")
endif()
set(HIPDNN_EP_GGML_SOURCE_DIR "${_llama_cpp_source_dir}/ggml")
if(NOT EXISTS "${HIPDNN_EP_GGML_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "llama.cpp's ggml is unavailable at ${HIPDNN_EP_GGML_SOURCE_DIR}")
endif()

set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(_ggml_saved_shared ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
add_subdirectory("${HIPDNN_EP_GGML_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ggml"
  EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_ggml_saved_shared})
foreach(_ggml_target ggml-base ggml-cpu ggml)
  set_target_properties(${_ggml_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()

add_library(NlohmannJson INTERFACE)
target_include_directories(NlohmannJson SYSTEM INTERFACE
  "${_llama_cpp_source_dir}/vendor")
