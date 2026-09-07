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
    <SOURCE_DIR>/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/libkernel.a
    <SOURCE_DIR>/${CHEZ_MACHINE}/lz4/lib/liblz4.a
    <SOURCE_DIR>/${CHEZ_MACHINE}/zlib/libz.a
)

ExternalProject_Get_Property(ChezScheme SOURCE_DIR)
set(ChezScheme_SOURCE_DIR "${SOURCE_DIR}")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ChezBootEmbedded.cpp
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/cmake/xxd.py
          --var petite_boot_data
          --output ${CMAKE_CURRENT_BINARY_DIR}/ChezBootPetite.cpp
          ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/petite.boot
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/cmake/xxd.py
          --var scheme_boot_data
          --output ${CMAKE_CURRENT_BINARY_DIR}/ChezBootScheme.cpp
          ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/scheme.boot
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/cmake/xxd.py
          --var print_operation_scm_data
          --output ${CMAKE_CURRENT_BINARY_DIR}/PrintOperationScm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/print-operation.scm
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/cmake/merge-embedded-files.py
          ${CMAKE_CURRENT_BINARY_DIR}/ChezBootEmbedded.cpp
          ${CMAKE_CURRENT_BINARY_DIR}/ChezBootPetite.cpp:${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/petite.boot:petite_boot_size
          ${CMAKE_CURRENT_BINARY_DIR}/ChezBootScheme.cpp:${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/scheme.boot:scheme_boot_size
          ${CMAKE_CURRENT_BINARY_DIR}/PrintOperationScm.cpp:${CMAKE_CURRENT_SOURCE_DIR}/print-operation.scm:print_operation_scm_size
  COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/ChezBootPetite.cpp ${CMAKE_CURRENT_BINARY_DIR}/ChezBootScheme.cpp ${CMAKE_CURRENT_BINARY_DIR}/PrintOperationScm.cpp
  DEPENDS ChezScheme
          ${CMAKE_CURRENT_SOURCE_DIR}/print-operation.scm
          ${CMAKE_SOURCE_DIR}/cmake/merge-embedded-files.py
  COMMENT "Embedding Chez boot files and Scheme code as C arrays"
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
  ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}
  ${ChezScheme_SOURCE_DIR}/boot/pb
)

add_dependencies(HipsrSchemePass ChezScheme)

target_link_libraries(HipsrSchemePass
  ChezBoot
  ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/libkernel.a
  ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/lz4/lib/liblz4.a
  ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/zlib/libz.a
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
