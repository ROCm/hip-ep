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
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/SchemeBindingsScm.h
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/cmake/xxd.py
          --var scheme_bindings_scm_data
          --output ${CMAKE_CURRENT_BINARY_DIR}/SchemeBindingsScm.h
          ${CMAKE_CURRENT_SOURCE_DIR}/SchemeBindings.scm
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/SchemeBindings.scm
  COMMENT "Embedding SchemeBindings.scm"
  VERBATIM
)

add_custom_target(ChezBootHeaders DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/ChezBootPetite.h
  ${CMAKE_CURRENT_BINARY_DIR}/ChezBootScheme.h
  ${CMAKE_CURRENT_BINARY_DIR}/SchemeBindingsScm.h
)
add_library(HipsrSchemePass
  SchemeBindings.cpp
  SchemeScriptPass.cpp
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

# Copy Scheme scripts to build tree for testing
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/PrintPass.scm
  ${CMAKE_BINARY_DIR}/lib/scheme/PrintPass.scm
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/SchemeBindings.scm
  ${CMAKE_BINARY_DIR}/lib/scheme/SchemeBindings.scm
  COPYONLY
)

# Copy rime library to build tree for testing
file(COPY ${CMAKE_SOURCE_DIR}/third_party/rime/rime
     DESTINATION ${CMAKE_BINARY_DIR}/lib/scheme)

# Install Scheme scripts and rime library to <prefix>/lib/scheme/
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/PrintPass.scm
  ${CMAKE_CURRENT_SOURCE_DIR}/SchemeBindings.scm
  DESTINATION lib/scheme
)

# Install rime library
install(DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/rime/rime
        DESTINATION lib/scheme
        FILES_MATCHING PATTERN "*.sls" PATTERN "*.scm")
