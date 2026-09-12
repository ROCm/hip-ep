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
add_library(HipsrSchemePass
  Runtime/SchemeBindings.cpp
  SchemeScriptPass.cpp
  SchemeConversionPass.cpp
)

target_include_directories(HipsrSchemePass PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${ChezScheme_BINARY_DIR}/${CHEZ_MACHINE}
  ${ChezScheme_SOURCE_DIR}/boot/pb
)

add_dependencies(HipsrSchemePass ChezScheme ChezBootHeaders SchemeLibraries)

target_link_libraries(HipsrSchemePass
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

# Set up Scheme library directory structure
set(SCHEME_LIB_DIR ${CMAKE_BINARY_DIR}/lib/scheme)

# Get the scheme compiler from ChezScheme build
ExternalProject_Get_Property(ChezScheme BINARY_DIR)
set(SCHEME_COMPILER ${ChezScheme_BINARY_DIR}/${CHEZ_MACHINE}/bin/${CHEZ_MACHINE}/scheme)

# Mirror source directory structure in build tree
file(MAKE_DIRECTORY ${SCHEME_LIB_DIR}/mlir)
file(MAKE_DIRECTORY ${SCHEME_LIB_DIR}/mlir/conversion)

# List of R6RS library files to compile
set(MLIR_FFI_SLS ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/mlir/ffi.sls)
set(MLIR_PATTERN_DSL_SLS ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/mlir/pattern-dsl.sls)
set(MLIR_CAST_SLS ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/mlir/conversion/cast.sls)

# Helper script for compiling libraries
set(COMPILE_LIBRARY_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/compile-library.sh)

# Compile R6RS libraries to .so files
add_custom_command(
  OUTPUT ${SCHEME_LIB_DIR}/mlir/ffi.so
  COMMAND bash "${COMPILE_LIBRARY_SCRIPT}"
          "${SCHEME_COMPILER}"
          "mlir ffi"
          "${CMAKE_CURRENT_SOURCE_DIR}/Runtime"
          "${CMAKE_CURRENT_SOURCE_DIR}/Runtime"
          "${SCHEME_LIB_DIR}/mlir/ffi.so"
  DEPENDS ChezScheme ${MLIR_FFI_SLS} ${COMPILE_LIBRARY_SCRIPT}
  COMMENT "Compiling (mlir ffi) library"
  VERBATIM
)

add_custom_command(
  OUTPUT ${SCHEME_LIB_DIR}/mlir/pattern-dsl.so
  COMMAND bash "${COMPILE_LIBRARY_SCRIPT}"
          "${SCHEME_COMPILER}"
          "mlir pattern-dsl"
          "${CMAKE_CURRENT_SOURCE_DIR}/Runtime:${CMAKE_SOURCE_DIR}/third_party/rime"
          "${CMAKE_CURRENT_SOURCE_DIR}/Runtime"
          "${SCHEME_LIB_DIR}/mlir/pattern-dsl.so"
  DEPENDS ChezScheme ${SCHEME_LIB_DIR}/mlir/ffi.so ${MLIR_PATTERN_DSL_SLS} ${COMPILE_LIBRARY_SCRIPT}
  COMMENT "Compiling (mlir pattern-dsl) library"
  VERBATIM
)

add_custom_command(
  OUTPUT ${SCHEME_LIB_DIR}/mlir/conversion/cast.so
  COMMAND bash "${COMPILE_LIBRARY_SCRIPT}"
          "${SCHEME_COMPILER}"
          "mlir conversion cast"
          "${CMAKE_CURRENT_SOURCE_DIR}/Runtime:${CMAKE_SOURCE_DIR}/third_party/rime"
          "${CMAKE_CURRENT_SOURCE_DIR}/Runtime"
          "${SCHEME_LIB_DIR}/mlir/conversion/cast.so"
  DEPENDS ChezScheme ${SCHEME_LIB_DIR}/mlir/pattern-dsl.so ${MLIR_CAST_SLS} ${COMPILE_LIBRARY_SCRIPT}
  COMMENT "Compiling (mlir conversion cast) library"
  VERBATIM
)

# Copy rime library (for loop macro)
file(COPY ${CMAKE_SOURCE_DIR}/third_party/rime/rime
     DESTINATION ${SCHEME_LIB_DIR})

# Target for all compiled Scheme libraries
add_custom_target(SchemeLibraries ALL
  DEPENDS
    ${SCHEME_LIB_DIR}/mlir/ffi.so
    ${SCHEME_LIB_DIR}/mlir/pattern-dsl.so
    ${SCHEME_LIB_DIR}/mlir/conversion/cast.so
)

# Also copy source .sls files for development
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/mlir
     DESTINATION ${SCHEME_LIB_DIR})

# Entry point scripts (not compiled, loaded at runtime)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Passes/CastConversion-r6rs.scm
  ${SCHEME_LIB_DIR}/CastConversion-r6rs.scm
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Passes/PrintPass.scm
  ${SCHEME_LIB_DIR}/PrintPass.scm
  COPYONLY
)

# Backward compatibility: legacy .scm files
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/SchemeBindings.scm
  ${SCHEME_LIB_DIR}/SchemeBindings.scm
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/PatternDSL.scm
  ${SCHEME_LIB_DIR}/PatternDSL.scm
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Passes/CastConversion.scm
  ${SCHEME_LIB_DIR}/CastConversion.scm
  COPYONLY
)

# Install Scheme scripts and rime library to <prefix>/lib/scheme/
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/Passes/PrintPass.scm
  ${CMAKE_CURRENT_SOURCE_DIR}/Passes/CastConversion.scm
  ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/SchemeBindings.scm
  ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/PatternDSL.scm
  DESTINATION lib/scheme
)

# Install rime library
install(DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/rime/rime
        DESTINATION lib/scheme
        FILES_MATCHING PATTERN "*.sls" PATTERN "*.scm")
