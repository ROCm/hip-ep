##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HASH_LIBRARY OBJECT
  ../3rd-party/hash-library/crc32.cpp
  ../3rd-party/hash-library/crc32.h
  ../3rd-party/hash-library/hash.h
  ../3rd-party/hash-library/hmac.h
  ../3rd-party/hash-library/keccak.cpp
  ../3rd-party/hash-library/keccak.h
  ../3rd-party/hash-library/md5.cpp
  ../3rd-party/hash-library/md5.h
  ../3rd-party/hash-library/sha1.cpp
  ../3rd-party/hash-library/sha1.h
  ../3rd-party/hash-library/sha256.cpp
  ../3rd-party/hash-library/sha256.h
  ../3rd-party/hash-library/sha3.cpp
  ../3rd-party/hash-library/sha3.h)

target_link_libraries(morphizen-core-static PRIVATE HASH_LIBRARY)
set_target_properties(HASH_LIBRARY PROPERTIES FOLDER Dependencies/hash-library)
