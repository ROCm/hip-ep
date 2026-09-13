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

# Compile all Scheme libraries in one shot - super fast!
set(COMPILE_ALL_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/compile-all-libraries.sh)
set(ALL_SCHEME_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/mlir/ffi.sls
  ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/mlir/pattern-dsl.sls
  ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/mlir/conversion/cast.sls
  ${CMAKE_CURRENT_SOURCE_DIR}/Passes/onnx-to-hipsr.sls
)

add_custom_command(
  OUTPUT
    ${SCHEME_LIB_DIR}/mlir/ffi.so
    ${SCHEME_LIB_DIR}/mlir/pattern-dsl.so
    ${SCHEME_LIB_DIR}/mlir/conversion/cast.so
    ${SCHEME_LIB_DIR}/onnx-to-hipsr.so
  COMMAND bash "${COMPILE_ALL_SCRIPT}"
          "${SCHEME_COMPILER}"
          "${CMAKE_CURRENT_SOURCE_DIR}"
          "${SCHEME_LIB_DIR}"
  DEPENDS ChezScheme ${ALL_SCHEME_SOURCES} ${COMPILE_ALL_SCRIPT}
  COMMENT "Compiling all Scheme libraries to bytecode"
  VERBATIM
)

# Target for all compiled Scheme libraries
add_custom_target(SchemeLibraries ALL
  DEPENDS
    ${SCHEME_LIB_DIR}/mlir/ffi.so
    ${SCHEME_LIB_DIR}/mlir/pattern-dsl.so
    ${SCHEME_LIB_DIR}/mlir/conversion/cast.so
    ${SCHEME_LIB_DIR}/onnx-to-hipsr.so
)

# Copy simple pass modules (these are just thin wrappers, not compiled)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Passes/printpass.sls
  ${SCHEME_LIB_DIR}/printpass.sls
  COPYONLY
)

# Install R6RS pass modules to <prefix>/lib/scheme/
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/Passes/printpass.sls
  DESTINATION lib/scheme
)

# Install compiled R6RS libraries (.so bytecode)
install(FILES
  ${SCHEME_LIB_DIR}/mlir/ffi.so
  ${SCHEME_LIB_DIR}/mlir/pattern-dsl.so
  ${SCHEME_LIB_DIR}/mlir/conversion/cast.so
  ${SCHEME_LIB_DIR}/onnx-to-hipsr.so
  DESTINATION lib/scheme/mlir
)

# Install rime library (dependency)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/rime/rime
        DESTINATION lib/scheme
        FILES_MATCHING PATTERN "*.sls" PATTERN "*.scm")
