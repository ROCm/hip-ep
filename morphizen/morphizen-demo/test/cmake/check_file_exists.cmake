##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
message(STATUS "Checking if file exists: ${VAR_FILE}")
if(EXISTS "${VAR_FILE}")
    message(STATUS "File exists: ${VAR_FILE}")
else()
    message(FATAL_ERROR "File does not exist: ${VAR_FILE}")
endif()
