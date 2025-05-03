##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-inline"
)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
string(APPEND CMAKE_CXX_FLAGS
  " -std=gnu++17 -Wall -Werror -Wconversion -pedantic -Wextra")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-parameter")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wconversion")
set(CMAKE_SHARED_LINKER_FLAGS
  "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-sign-conversion")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-nested-anon-types")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-invalid-utf8")
  ## TODO: it is not good to disable the following warnings.
  string(APPEND CMAKE_CXX_FLAGS " -Wno-shorten-64-to-32")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-implicit-int-conversion")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-self-assign")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-delete-non-abstract-non-virtual-dtor")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-attributes")
endif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
