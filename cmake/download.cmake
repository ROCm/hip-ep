##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
message(STATUS "download ${URL} to ${FILE}")
set(args "${URL}" "${FILE}")
if(EXPECTED_MD5)
    list(APPEND args "EXPECTED_MD5" "${EXPECTED_MD5}")
endif()
file(DOWNLOAD ${args} TLS_VERIFY OFF)
if(PATCH_FILES)
    foreach(patch_file ${PATCH_FILES})
        execute_process(COMMAND "${Patch_EXECUTABLE}" -i "${patch_file}" -p1)
    endforeach()
endif()
