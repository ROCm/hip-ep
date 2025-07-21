##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.16)

project(vaip_io
    VERSION 1.0.0
    DESCRIPTION "VAIP IO - Stream I/O utilities for VAIP"
    LANGUAGES CXX
)

add_library(vaip_io STATIC
    src/vaip_io.cpp
)

# Specify include directories
target_include_directories(vaip_io
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(vaip_io
    PUBLIC
       glog::glog
)

# Add include directories for dependencies
target_include_directories(vaip_io PRIVATE ${GLOG_INCLUDE_DIRS})

# Set compile features
target_compile_features(vaip_io PUBLIC cxx_std_17)
target_compile_options(vaip_io PRIVATE ${MORPHIZEN_COMPILER_OPTIONS})


# Create an alias for consistent naming
add_library(vaip::io ALIAS vaip_io)
