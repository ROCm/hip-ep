# Chez Scheme Integration for HipSR Dialect
# Builds Chez with rime embedded in custom boot file

include(ExternalProject)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Step 1: Build Chez Scheme
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
    <SOURCE_DIR>/bin/ti3le/scheme
    <SOURCE_DIR>/boot/ti3le/petite.boot
    <SOURCE_DIR>/boot/ti3le/scheme.boot
)

ExternalProject_Get_Property(ChezScheme SOURCE_DIR)
set(ChezScheme_SOURCE_DIR "${SOURCE_DIR}")

# Step 2: Create custom boot file with rime
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hip-patterns.boot
  COMMAND ${ChezScheme_SOURCE_DIR}/bin/ti3le/scheme
          --script ${CMAKE_SOURCE_DIR}/cmake/build-chez-boot.scm
          ${ChezScheme_SOURCE_DIR}/boot/ti3le/petite.boot
          ${ChezScheme_SOURCE_DIR}/boot/ti3le/scheme.boot
          ${CMAKE_CURRENT_BINARY_DIR}/hip-patterns.boot
  DEPENDS ChezScheme
          ${CMAKE_SOURCE_DIR}/cmake/build-chez-boot.scm
  COMMENT "Compiling rime into custom boot file"
  VERBATIM
)

# Step 3: Embed boot file as C array (using existing xxd.py)
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
  COMMENT "Embedding Chez boot file (with rime) as C array"
  VERBATIM
)

# Step 4: Create library with embedded boot
add_library(ChezBoot STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/ChezBootEmbedded.cpp
)
set_target_properties(ChezBoot PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LINKER_LANGUAGE CXX
)

# Step 5: Create Scheme FFI and pass library
add_library(HipsrSchemePass
  SchemeRuntime.cpp
  SchemePrintPass.cpp
)

target_include_directories(HipsrSchemePass PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${ChezScheme_SOURCE_DIR}/ta6le
)

target_link_libraries(HipsrSchemePass
  ChezBoot
  MLIRPass
  MLIRSupport
  MLIRIR
  HipsrDialect
)
