##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.16)
project(morphizen-utils
    VERSION 1.0.0
    DESCRIPTION "Utility library for MorphiZen project"
    LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/../morphizen-core/cmake/find_onnxruntime.cmake)

# Library source files
set(MORPHIZEN_UTILS_SOURCES
    src/morphizen_plugin.cpp
    src/cleanup.cpp
)

if(WIN32)
  list(APPEND MORPHIZEN_UTILS_SOURCES src/morphizen_plugin_win.cpp)
else()
  list(APPEND MORPHIZEN_UTILS_SOURCES src/morphizen_plugin_lnx.cpp)
endif()
# Library header files
set(MORPHIZEN_UTILS_HEADERS
    include/morphizen-utils/morphizen-utils.hpp
    include/morphizen-utils/weak_refs.hpp
    include/morphizen-utils/morphizen_plugin.hpp
    include/morphizen-utils/cleanup.hpp
)

# Create the library
add_library(morphizen-utils STATIC ${MORPHIZEN_UTILS_SOURCES} ${MORPHIZEN_UTILS_HEADERS})

# Set target properties
set_target_properties(morphizen-utils PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${MORPHIZEN_UTILS_HEADERS}"
)

# Include directories
target_include_directories(morphizen-utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${MORPHIZEN_ORT_API_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compiler-specific options
message("MORPHIZEN_COMPILER_OPTIONS is ${MORPHIZEN_COMPILER_OPTIONS}")
target_compile_options(morphizen-utils PRIVATE ${MORPHIZEN_COMPILER_OPTIONS})
target_link_libraries(morphizen-utils
    PUBLIC morphizen::foundation
    PRIVATE glog::glog
)

# Optional: Enable testing
option(MORPHIZEN_UTILS_BUILD_TESTS "Build tests for morphizen-utils" OFF)

if(MORPHIZEN_UTILS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
# DO NOT install the library by default because it is not ready for public use yet.
# MorphiZen is still an uncommon project which requires the clients to use fetchContent
# instead.

# # Install configuration
# include(GNUInstallDirs)

# install(TARGETS morphizen-utils
#     EXPORT morphizen-utils-targets
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#     PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/morphizen-utils
# )

# install(EXPORT morphizen-utils-targets
#     FILE morphizen-utils-config.cmake
#     NAMESPACE morphizen::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/morphizen-utils
# )

# Create a simple example
if(MORPHIZEN_UTILS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
