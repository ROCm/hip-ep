##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# hash-library is now managed via FetchContent in cmake/deps.cmake
# Just create the OBJECT library from the fetched sources
add_library(HASH_LIBRARY OBJECT
  ${hash-library_SOURCE_DIR}/crc32.cpp
  ${hash-library_SOURCE_DIR}/crc32.h
  ${hash-library_SOURCE_DIR}/hash.h
  ${hash-library_SOURCE_DIR}/hmac.h
  ${hash-library_SOURCE_DIR}/keccak.cpp
  ${hash-library_SOURCE_DIR}/keccak.h
  ${hash-library_SOURCE_DIR}/md5.cpp
  ${hash-library_SOURCE_DIR}/md5.h
  ${hash-library_SOURCE_DIR}/sha1.cpp
  ${hash-library_SOURCE_DIR}/sha1.h
  ${hash-library_SOURCE_DIR}/sha256.cpp
  ${hash-library_SOURCE_DIR}/sha256.h
  ${hash-library_SOURCE_DIR}/sha3.cpp
  ${hash-library_SOURCE_DIR}/sha3.h)

target_include_directories(HASH_LIBRARY PUBLIC ${hash-library_SOURCE_DIR})
target_link_libraries(morphizen-core-static PRIVATE HASH_LIBRARY)
set_target_properties(HASH_LIBRARY PROPERTIES FOLDER Dependencies/hash-library)
