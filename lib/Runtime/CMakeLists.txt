##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

file(GLOB RUNTIME_SRCS "ops_runtime/*.cpp")
add_library(hip_runtime_static STATIC ${RUNTIME_SRCS})
set_target_properties(hip_runtime_static PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)
target_compile_definitions(hip_runtime_static PRIVATE __HIP_PLATFORM_AMD__ MIOPEN_BETA_API)
if(DEFINED ENV{THEROCK_DIST})
  target_include_directories(hip_runtime_static PRIVATE $ENV{THEROCK_DIST}/include)
endif()
