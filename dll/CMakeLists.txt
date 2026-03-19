##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building hip-compiler.dll")

# Create the DLL target
add_library(hip-compiler-dll SHARED
    dll_main.cpp
    hip-compiler.def
)

# Link against CInterface library (which pulls in all dependencies)
target_link_libraries(hip-compiler-dll PRIVATE
    HipCInterface
)

# Include directories for C API headers
target_include_directories(hip-compiler-dll PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Set output name (no prefix on Windows)
# Don't override RUNTIME_OUTPUT_DIRECTORY - let it use CMAKE_RUNTIME_OUTPUT_DIRECTORY variable
set_target_properties(hip-compiler-dll PROPERTIES
    OUTPUT_NAME "hip-compiler"
)

# Install DLL and import library
install(TARGETS hip-compiler-dll
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Install C API headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/hip
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

message(STATUS "hip-compiler.dll configuration complete")
