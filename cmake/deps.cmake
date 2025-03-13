#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
include(FetchContent)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/deps.txt VAIP_DEPS_LIST)
file(READ "${CMAKE_CURRENT_LIST_DIR}/dep.h.inc.in" VAIP_DEP_H_INC_IN)
set(VAIP_DEP_H_INC "")
foreach(VAIP_DEP IN LISTS VAIP_DEPS_LIST)
  message("VAIP_DEP = ${VAIP_DEP}")
  # Lines start with "#" are comments
  if(NOT VAIP_DEP MATCHES "^#")
    # The first column is name
    list(POP_FRONT VAIP_DEP VAIP_DEP_NAME)
    # The second column is URL
    # The URL below may be a local file path or an HTTPS URL
    list(POP_FRONT VAIP_DEP VAIP_DEP_URL)
    set(DEP_URL_${VAIP_DEP_NAME} ${VAIP_DEP_URL})
    # The third column is SHA1 hash value
    set(DEP_SHA1_${VAIP_DEP_NAME} ${VAIP_DEP})
    string(CONFIGURE "${VAIP_DEP_H_INC_IN}" _tmp @ONLY)
    string(APPEND VAIP_DEP_H_INC "${_tmp}")
  endif()
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/vaip_deps.inc.h" "${VAIP_DEP_H_INC}")

find_package(Eigen3 QUIET)
if(TARGET Eigen3::Eigen)
  get_target_property(TMP Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "found Eigen3 at ${TMP}")
else()
  message(STATUS "cannot find_package(Eigen3), fetch it from ${DEP_URL_eigen}")
  FetchContent_Declare(
    Eigen3
    URL ${DEP_URL_eigen}
    URL_HASH SHA1=${DEP_SHA1_eigen}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    OVERRIDE_FIND_PACKAGE)
  find_package(Eigen3 REQUIRED)
endif()

find_package(Microsoft.GSL QUIET)
if(NOT TARGET Microsoft.GSL::GSL)
  message(STATUS "cannot find_package(Microsoft.GSL), fetch it from ${DEP_URL_microsoft_gsl}")
  FetchContent_Declare(
    Microsoft.GSL
    URL ${DEP_URL_microsoft_gsl}
    URL_HASH SHA1=${DEP_SHA1_microsoft_gsl}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE
  )
  find_package(Microsoft.GSL REQUIRED)
endif()

find_package(GTest QUIET)
if(GTest_FOUND)
  message(STATUS "found find_package(GTest)")
else()
  message(STATUS "fetch GTest from ${DEP_URL_GTest}")
   FetchContent_Declare(
    GTest
    GIT_REPOSITORY ${DEP_URL_GTest}
    GIT_TAG ${DEP_SHA1_GTest}
    GIT_SHALLOW TRUE
    CMAKE_ARGS -Dgtest_force_shared_crt=ON
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE
  )
  find_package(GTest REQUIRED)
endif()

set(WITH_GFLAGS OFF CACHE BOOL "disable WITH_GFLAGS for glog")
find_package(glog QUIET)
if(glog_FOUND)
  get_target_property(TMP glog::glog LOCATION)
  message(STATUS "found glog at ${TMP}")
else()
  message(STATUS "cannot find_package(glog), fetch it from ${DEP_URL_glog}")
  FetchContent_Declare(
    glog
    URL ${DEP_URL_glog}
    URL_HASH SHA1=${DEP_SHA1_glog}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE
  )
  find_package(glog REQUIRED)
endif()

set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "use static zip")
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
  message(STATUS "found find_package(ZLIB)")
else()
  message(STATUS "cannot find_package(ZLIB), fetch it from ${DEP_URL_zlib}")
  FetchContent_Declare(
    ZLIB
    GIT_REPOSITORY ${DEP_URL_zlib}
    GIT_TAG ${DEP_SHA1_zlib}
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE)
  find_package(ZLIB REQUIRED)
  if(NOT TARGET zlibstatic)
    message(STATUS "----- not find target zlibstatic")
  else()
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
  endif()
endif()


set(Protobuf_USE_STATIC_LIBS ON CACHE BOOL "use static protobuf")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "disable protobuf tests")
set(protobuf_WITH_ZLIB OFF CACHE BOOL "disable zlib for protobuf")
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "disable protobuf build shared libs")
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "disable protobuf examples")
find_package(Protobuf QUIET)
if(TARGET protobuf::libprotobuf)
  get_target_property(TMP protobuf::libprotobuf LOCATION)
  message(STATUS "found protobuf at ${TMP}")
else()
  message(STATUS "cannot find_package(Protobuf), fetch it from ${DEP_URL_protobuf}")
  FetchContent_Declare(
    Protobuf
    URL ${DEP_URL_protobuf}
    URL_HASH SHA1=${DEP_SHA1_protobuf}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE)
  find_package(Protobuf REQUIRED)
endif()

find_package(nlohmann_json QUIET)
if(TARGET  nlohmann_json::nlohmann_json)
  get_target_property(TMP nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "found nlohmann_json at ${TMP}")
else()
  message(STATUS "cannot find_package(nlohmann_json), fetch it from ${DEP_URL_json}")
  FetchContent_Declare(
    nlohmann_json
    URL ${DEP_URL_json}
    URL_HASH SHA1=${DEP_SHA1_json}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE)
  find_package(nlohmann_json REQUIRED)
endif()

## in order to build it, we need to run some python scripts to
## generate some source code.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
