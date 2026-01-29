##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.16)

project(morphizen-io
    VERSION 1.0.0
    DESCRIPTION "MorphiZen IO - Stream I/O utilities for MorphiZen"
    LANGUAGES CXX
)

add_library(morphizen-io STATIC
    src/morphizen_io.cpp
)

# Specify include directories
target_include_directories(morphizen-io
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(morphizen-io
    PUBLIC
       glog::glog
)

# Add include directories for dependencies
target_include_directories(morphizen-io PRIVATE ${GLOG_INCLUDE_DIRS})

# Set compile features
target_compile_features(morphizen-io PUBLIC cxx_std_17)
target_compile_options(morphizen-io PRIVATE ${MORPHIZEN_COMPILER_OPTIONS})


# Create aliases for consistent naming
add_library(morphizen::io ALIAS morphizen-io)
# Backward compatibility alias (legacy target name)
add_library(vaip_io ALIAS morphizen-io)
