##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(SOURCES
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


target_sources(morphizen-core-static PRIVATE ${SOURCES})

# Apply compile options only to those files
foreach(source_file ${SOURCES})
  if(MSVC)
    set_source_files_properties(${source_file} PROPERTIES COMPILE_FLAGS "/W3")
  else(MSVC)
    set_source_files_properties(${source_file} PROPERTIES COMPILE_FLAGS "-Wno-error")
  endif(MSVC)
endforeach()
