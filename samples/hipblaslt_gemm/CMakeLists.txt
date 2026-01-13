cmake_minimum_required(VERSION 3.21)

project(sample_hipblaslt_gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TheRock SDK location
if(NOT DEFINED ENV{THEROCK_DIST} AND NOT DEFINED THEROCK_DIST)
    set(THEROCK_DIST "C:/Develop/m/dist/therock")
else()
    if(DEFINED ENV{THEROCK_DIST})
        set(THEROCK_DIST $ENV{THEROCK_DIST})
    endif()
endif()

message(STATUS "Using TheRock at: ${THEROCK_DIST}")

# Set HIP_PLATFORM if not set
if(NOT DEFINED ENV{HIP_PLATFORM})
    set(ENV{HIP_PLATFORM} "amd")
endif()

# Add TheRock to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")

# Find HIP package
find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED)

message(STATUS "Found HIP: ${hip_DIR}")
message(STATUS "Found hipBLASLt: ${hipblaslt_DIR}")

# Create executable
add_executable(sample_hipblaslt_gemm sample_hipblaslt_gemm.cpp)

# Set include directories
target_include_directories(sample_hipblaslt_gemm PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${THEROCK_DIST}/include"
)

# Link libraries
target_link_libraries(sample_hipblaslt_gemm PRIVATE hip::device roc::hipblaslt)

# For Windows, we may need to set additional options
if(WIN32)
    target_compile_definitions(sample_hipblaslt_gemm PRIVATE _USE_MATH_DEFINES)
endif()
