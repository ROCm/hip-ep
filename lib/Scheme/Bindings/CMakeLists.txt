##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_subdirectory(Dialects)

# ChezScheme properties set by parent Interpreter/CMakeLists.txt
set(CHEZ_MACHINE_LOCAL "${CHEZ_MACHINE}")

add_library(HipsrSchemeRuntime
  ${CMAKE_CURRENT_SOURCE_DIR}/../Interpreter/ChezSchemeInterpreter.cpp
  MainModule.cpp
  Core.cpp
  Logging.cpp
  Conversion.cpp
  Dialects/Tensor.cpp
  Dialects/Shape.cpp
  Dialects/Hipsr.cpp
  Dialects/Onnx.cpp
)

target_include_directories(HipsrSchemeRuntime
  PUBLIC
    ${ChezScheme_BINARY_DIR}/${CHEZ_MACHINE}
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ChezBootHeaders_BINARY_DIR}
    ${ChezScheme_SOURCE_DIR}/boot/pb
)

# Scheme library paths: source .sls files live in scheme/ at project root
target_compile_definitions(HipsrSchemeRuntime PRIVATE
  SCHEME_LIBRARIES_DIR="${CMAKE_SOURCE_DIR}/scheme"
  SCHEME_BINARY_DIR="${CMAKE_BINARY_DIR}/scheme"
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
