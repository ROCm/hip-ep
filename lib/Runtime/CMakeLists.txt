##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

file(GLOB RUNTIME_SRCS "ops_runtime/*.cpp")
add_library(hip_runtime_static STATIC ${RUNTIME_SRCS})
set_target_properties(hip_runtime_static PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
target_compile_definitions(hip_runtime_static PRIVATE __HIP_PLATFORM_AMD__ MIOPEN_BETA_API)
if(DEFINED ENV{THEROCK_DIST})
  target_include_directories(hip_runtime_static PRIVATE $ENV{THEROCK_DIST}/include)
endif()
