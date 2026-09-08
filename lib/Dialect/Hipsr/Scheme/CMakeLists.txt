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

set(CHEZ_EMBEDDED_FILES
  "ChezBootPetite:petite_boot_data:${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/petite.boot:ChezScheme"
  "ChezBootScheme:scheme_boot_data:${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}/boot/${CHEZ_MACHINE}/scheme.boot:ChezScheme"
  "PrintOperationScm:print_operation_scm_data:${CMAKE_CURRENT_SOURCE_DIR}/print-operation.scm:${CMAKE_CURRENT_SOURCE_DIR}/print-operation.scm"
)

set(CHEZ_HEADER_FILES "")
foreach(ENTRY ${CHEZ_EMBEDDED_FILES})
  string(REPLACE ":" ";" PARTS ${ENTRY})
  list(GET PARTS 0 HEADER_NAME)
  list(GET PARTS 1 VAR_NAME)
  list(GET PARTS 2 INPUT_FILE)
  list(GET PARTS 3 DEPENDS_TARGET)

  set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${HEADER_NAME}.h)
  list(APPEND CHEZ_HEADER_FILES ${OUTPUT_FILE})

  add_custom_command(
    OUTPUT ${OUTPUT_FILE}
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/cmake/xxd.py
            --var ${VAR_NAME}
            --output ${OUTPUT_FILE}
            ${INPUT_FILE}
    DEPENDS ${DEPENDS_TARGET}
    COMMENT "Embedding ${INPUT_FILE}"
    VERBATIM
  )
endforeach()

add_custom_target(ChezBootHeaders DEPENDS ${CHEZ_HEADER_FILES})
add_library(HipsrSchemePass
  SchemeRuntime.cpp
  SchemePrintPass.cpp
)

target_include_directories(HipsrSchemePass PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${ChezScheme_SOURCE_DIR}/${CHEZ_MACHINE}
  ${ChezScheme_SOURCE_DIR}/boot/pb
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
