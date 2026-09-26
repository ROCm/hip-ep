##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

include(ExternalProject)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Chez Scheme integration requires 64-bit architecture")
endif()

if(WIN32)
  set(CHEZ_MACHINE "ta6nt")
else()
  set(CHEZ_MACHINE "ta6le")
endif()

message(STATUS "Chez Scheme machine type: ${CHEZ_MACHINE}")

ExternalProject_Add(ChezScheme
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/ChezScheme
  BINARY_DIR ${CMAKE_BINARY_DIR}/ChezScheme-build
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E chdir <BINARY_DIR>
    ${CMAKE_SOURCE_DIR}/third_party/ChezScheme/configure
    --threads
    --disable-x11
    --machine=${CHEZ_MACHINE}
  BUILD_COMMAND ${CMAKE_COMMAND} -E chdir <BINARY_DIR> make -j8
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS
    <BINARY_DIR>/${CHEZ_MACHINE}/bin/${CHEZ_MACHINE}/scheme
    <BINARY_DIR>/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/petite.boot
    <BINARY_DIR>/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/scheme.boot
    <BINARY_DIR>/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/libkernel.a
    <BINARY_DIR>/${CHEZ_MACHINE}/lz4/lib/liblz4.a
    <BINARY_DIR>/${CHEZ_MACHINE}/zlib/libz.a
)

ExternalProject_Get_Property(ChezScheme SOURCE_DIR BINARY_DIR)
set(ChezScheme_SOURCE_DIR "${SOURCE_DIR}")
set(ChezScheme_BINARY_DIR "${BINARY_DIR}")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ChezBootPetite.h
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/cmake/xxd.py
          --var petite_boot_data
          --output ${CMAKE_CURRENT_BINARY_DIR}/ChezBootPetite.h
          ${ChezScheme_BINARY_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/petite.boot
  DEPENDS ChezScheme
  COMMENT "Embedding petite.boot"
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ChezBootScheme.h
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/cmake/xxd.py
          --var scheme_boot_data
          --output ${CMAKE_CURRENT_BINARY_DIR}/ChezBootScheme.h
          ${ChezScheme_BINARY_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/scheme.boot
  DEPENDS ChezScheme
  COMMENT "Embedding scheme.boot"
  VERBATIM
)

add_custom_target(ChezBootHeaders DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/ChezBootPetite.h
  ${CMAKE_CURRENT_BINARY_DIR}/ChezBootScheme.h
)
add_library(HipsrSchemeRuntime
  Runtime/ChezSchemeInterpreter.cpp
  Runtime/SchemeMlirBindings.cpp
)

target_include_directories(HipsrSchemeRuntime
  PUBLIC
    ${ChezScheme_BINARY_DIR}/${CHEZ_MACHINE}
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ChezScheme_SOURCE_DIR}/boot/pb
)

# Define paths for Scheme library directories (source .sls files and binary .so files)
target_compile_definitions(HipsrSchemeRuntime PRIVATE
  SCHEME_LIBRARIES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/libraries"
  SCHEME_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}/libraries"
  RIME_DIR="${CMAKE_SOURCE_DIR}/third_party/rime"
)

add_dependencies(HipsrSchemeRuntime ChezScheme ChezBootHeaders)

target_link_libraries(HipsrSchemeRuntime
  ${ChezScheme_BINARY_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/libkernel.a
  ${ChezScheme_BINARY_DIR}/${CHEZ_MACHINE}/lz4/lib/liblz4.a
  ${ChezScheme_BINARY_DIR}/${CHEZ_MACHINE}/zlib/libz.a
  MLIRPass
  MLIRSupport
  MLIRIR
  HipsrDialectIR
  ${CMAKE_DL_LIBS}
  m
  pthread
  ncurses
  util
)

# DO NOT compile .sls files - use them as source
# Scheme libraries will be loaded directly from source by the runtime
