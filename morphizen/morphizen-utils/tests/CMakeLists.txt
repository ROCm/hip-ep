##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Test CMakeLists for morphizen-utils
cmake_minimum_required(VERSION 3.16)

# Create test executable
add_executable(morphizen-utils-test test_main.cpp)

# Link to the main library
target_link_libraries(morphizen-utils-test morphizen-utils)

# Add test to CTest
add_test(NAME morphizen-utils-basic-test COMMAND morphizen-utils-test)
