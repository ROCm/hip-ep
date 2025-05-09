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

set(MORPHIZEN_COMPILER_OPTIONS
  -Wall -Werror -Wconversion -pedantic
  -Wextra -fPIC
  CACHE STRING "Compiler options for Morphizen"
)

set(MORPHIZEN_LINKER_OPTIONS
  "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined"
  CACHE STRING "Linker options for Morphizen")
