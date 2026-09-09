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
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ChezBootPetite.h
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/cmake/xxd.py
          --var petite_boot_data
          --output ${CMAKE_CURRENT_BINARY_DIR}/ChezBootPetite.h
          ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/petite.boot
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
          ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/scheme.boot
  DEPENDS ChezScheme
  COMMENT "Embedding scheme.boot"
  VERBATIM
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/PrintOperationScm.h
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/cmake/xxd.py
          --var print_operation_scm_data
          --output ${CMAKE_CURRENT_BINARY_DIR}/PrintOperationScm.h
          ${CMAKE_CURRENT_SOURCE_DIR}/scheme-mlir-bridge.scm
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scheme-mlir-bridge.scm
  COMMENT "Embedding scheme-mlir-bridge.scm"
  VERBATIM
)

add_custom_target(ChezBootHeaders DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/ChezBootPetite.h
  ${CMAKE_CURRENT_BINARY_DIR}/ChezBootScheme.h
  ${CMAKE_CURRENT_BINARY_DIR}/PrintOperationScm.h
)
add_library(HipsrSchemePass
  SchemeRuntime.cpp
  SchemeScriptPass.cpp
)

target_include_directories(HipsrSchemePass PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}
  ${ChezScheme_SOURCE_DIR}/boot/pb
)

target_compile_definitions(HipsrSchemePass PRIVATE
  SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_dependencies(HipsrSchemePass ChezScheme ChezBootHeaders)

target_link_libraries(HipsrSchemePass
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
