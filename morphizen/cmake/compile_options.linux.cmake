##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-inline"
)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

set(CMAKE_SHARED_LINKER_FLAGS
  "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(MORPHIZEN_COMPILER_OPTIONS
  -Wall -Werror -Wconversion -pedantic
  -Wextra -fPIC
  CACHE STRING "Compiler options for Morphizen"
)

# Clang (e.g. clang-20) enables many diagnostics gcc does not, and this tree
# builds -Werror. These fire throughout vendored morphizen code that compiles
# clean under its intended gcc toolchain. Rather than enumerate every one,
# demote ALL warnings to non-fatal for Clang only (real hard errors stay
# fatal), so a Clang-configured build links without touching gcc/CI behavior or
# the vendored sources.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND MORPHIZEN_COMPILER_OPTIONS -Wno-error)
  set(MORPHIZEN_COMPILER_OPTIONS "${MORPHIZEN_COMPILER_OPTIONS}"
      CACHE STRING "Compiler options for Morphizen" FORCE)
endif()

set(MORPHIZEN_LINKER_OPTIONS
  "-ggdb"
  "-Wl,--no-undefined"
  CACHE STRING "Linker options for Morphizen")
