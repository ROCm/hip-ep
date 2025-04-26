#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#
set(FETCHCONTENT_QUIET TRUE CACHE BOOL "enable fetchcontent quiet")
include(FetchContent)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/deps.txt VAIP_DEPS_LIST)
file(READ "${CMAKE_CURRENT_LIST_DIR}/dep.h.inc.in" VAIP_DEP_H_INC_IN)
set(VAIP_DEP_H_INC "")
foreach(VAIP_DEP IN LISTS VAIP_DEPS_LIST)
  # Lines start with "#" are comments
  if(NOT VAIP_DEP MATCHES "^#")
    message(STATUS "VAIP_DEP = ${VAIP_DEP}")
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

if(morphizen_ENABLE_UNIT_TEST)
  find_package(GTest CONFIG QUIET)
  if(TARGET GTest::gtest)
    get_target_property(TMP GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "found find_package(GTest) at ${TMP}")
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
endif()

set(WITH_GFLAGS OFF CACHE BOOL "disable WITH_GFLAGS for glog")
find_package(glog CONFIG QUIET)
if(TARGET glog::glog)
  get_target_property(TMP glog::glog INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "found glog at ${TMP} ")
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


# must use static zlib on windows,even if BUILD_SHARED_LIBS is undfined or OFF,
# because we assume static msvc runtime is used
# because file opened in one dll cannot be used in another dll
# see unit-test PassContextTest.TestCompress
# NOTE: BUILD_SHARED_LIBS might not be defined
if(MSVC)
  set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "use static zip")
elseif(NOT ${BUILD_SHARED_LIBS})
  set(ZLIB_USE_STATIC_LIBS OFF CACHE BOOL "use static zip")
endif()
# todo: try to find prebuilt zlib
if(TARGET ZLIB::ZLIB)
  get_target_property(TMP ZLIB::ZLIB INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "found ZLIB at ${TMP}")
else()
  message(STATUS "cannot find_package(ZLIB), fetch it from ${DEP_URL_zlib}")
  FetchContent_Declare(
      ZLIB
      GIT_REPOSITORY ${DEP_URL_zlib}
      GIT_TAG ${DEP_SHA1_zlib}
      GIT_SHALLOW TRUE
      EXCLUDE_FROM_ALL
      OVERRIDE_FIND_PACKAGE
      PATCH_COMMAND
          ${CMAKE_COMMAND} -E echo "Checking if patch is already applied..." &&
          git apply --check "${CMAKE_CURRENT_SOURCE_DIR}/cmake/zlib.patch" -q &&
          git apply "${CMAKE_CURRENT_SOURCE_DIR}/cmake/zlib.patch" ||
          ${CMAKE_COMMAND} -E echo "Patch already applied or failed to apply"
  )
  find_package(ZLIB REQUIRED)
  if(ZLIB_USE_STATIC_LIBS)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
    target_include_directories(zlibstatic PUBLIC ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
    set_target_properties(zlibstatic PROPERTIES FOLDER morphizen/deps)
    message(STATUS "found static ZLIB at ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR}")
  else()
    add_library(ZLIB::ZLIB ALIAS zlib)
    target_include_directories(zlib PRIVATE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
    set_target_properties(zlib PROPERTIES FOLDER morphizen/deps)
    message(STATUS "found dynamic ZLIB at ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR}")
  endif()
  # TODO: I don't know why, the following line does not work we have
  # to set the include path explicitly in vaip_core_static

  # target_link_directories(zlibstatic PUBLIC ${zlib_BINARY_DIR} ${zlib_SOURCE_DIR})
endif()

if(NOT MSVC_RUNTIME_LIBRARY)
  set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "use dynamic msvc runtime for protobuf by default, /MD")
elseif(${MSVC_RUNTIME_LIBRARY} MATCHES "*DLL*")
  set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "use dynamic msvc runtime for protobuf, /MD")
else()
  set(protobuf_MSVC_STATIC_RUNTIME ON CACHE BOOL "use static msvc runtime for protobuf, /MT")
endif()
set(protobuf_BUILD_TESTS OFF CACHE BOOL "disable protobuf tests")
# TODO: enable ZLIB
set(protobuf_WITH_ZLIB OFF CACHE BOOL "disable zlib for protobuf")
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "disable protobuf build shared libs")
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "disable protobuf examples")
## it is error-prone to use MODULE mode to find protobuf,
## Protobuf_USE_STATIC_LIBS must be defined.
find_package(Protobuf CONFIG QUIET)
if(TARGET protobuf::libprotobuf)
  get_target_property(TMP protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
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
if(BUILD_PYTHON)
  message(STATUS "find_package(Python3) both for  python scripts and for python embedding")
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED CONFIG)
  if(TARGET pybind11::embed)
    get_target_property(TMP pybind11::embed INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "found pybind11 at ${TMP}")
  else()
    message(STATUS "cannot find_package(pybind11), fetch it from ${DEP_URL_pybind11}")
    FetchContent_Declare(
      pybind11
      URL ${DEP_URL_pybind11}
      URL_HASH SHA1=${DEP_SHA1_pybind11}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      OVERRIDE_FIND_PACKAGE)
    find_package(pybind11 REQUIRED)
  endif()
else()
  message(STATUS "find_package(Python3) for python scripts")
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

if(NOT TARGET Python3::Interpreter)
  message(FATAL_ERROR "Python3::Interpreter not found")
endif(NOT TARGET Python3::Interpreter)
