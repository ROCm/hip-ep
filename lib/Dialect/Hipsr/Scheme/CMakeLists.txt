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
  CONFIGURE_COMMAND ${CMAKE_SOURCE_DIR}/third_party/ChezScheme/configure
    --threads
    --disable-x11
    --installprefix=<INSTALL_DIR>
  BUILD_COMMAND make -j8
  INSTALL_COMMAND ""
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS
    <SOURCE_DIR>/${CHEZ_MACHINE}/bin/${CHEZ_MACHINE}/scheme
    <SOURCE_DIR>/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/petite.boot
    <SOURCE_DIR>/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/scheme.boot
)

ExternalProject_Get_Property(ChezScheme SOURCE_DIR)
set(ChezScheme_SOURCE_DIR "${SOURCE_DIR}")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hip-patterns.boot
  COMMAND ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/bin/${CHEZ_MACHINE}/scheme
          --script ${CMAKE_SOURCE_DIR}/cmake/build-chez-boot.scm
          ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/petite.boot
          ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/scheme.boot
          ${CMAKE_CURRENT_BINARY_DIR}/hip-patterns.boot
  DEPENDS ChezScheme
          ${CMAKE_SOURCE_DIR}/cmake/build-chez-boot.scm
  COMMENT "Creating custom boot file for ${CHEZ_MACHINE}"
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ChezBootEmbedded.cpp
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/cmake/xxd.py
          --var chez_boot_data
          --output ${CMAKE_CURRENT_BINARY_DIR}/ChezBootEmbedded.cpp.tmp
          ${CMAKE_CURRENT_BINARY_DIR}/hip-patterns.boot
  COMMAND ${Python3_EXECUTABLE} -c
          "import os; data = open('${CMAKE_CURRENT_BINARY_DIR}/ChezBootEmbedded.cpp.tmp').read(); size = os.path.getsize('${CMAKE_CURRENT_BINARY_DIR}/hip-patterns.boot'); data = data.replace('static const unsigned char', 'extern \"C\" const unsigned char'); data = '#include <cstddef>\\n' + data + f'\\nextern \"C\" const size_t chez_boot_size = {size};\\n'; open('${CMAKE_CURRENT_BINARY_DIR}/ChezBootEmbedded.cpp', 'w').write(data)"
  COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/ChezBootEmbedded.cpp.tmp
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hip-patterns.boot
  COMMENT "Embedding Chez boot file as C array"
  VERBATIM
)

add_library(ChezBoot STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/ChezBootEmbedded.cpp
)
set_target_properties(ChezBoot PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LINKER_LANGUAGE CXX
)

add_library(HipsrSchemePass
  SchemeRuntime.cpp
  SchemePrintPass.cpp
)

target_include_directories(HipsrSchemePass PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

add_dependencies(HipsrSchemePass ChezScheme)

target_link_libraries(HipsrSchemePass
  ChezBoot
  MLIRPass
  MLIRSupport
  MLIRIR
  HipsrDialectIR
  ${CMAKE_DL_LIBS}
  m
  pthread
)
