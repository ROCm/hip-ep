##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_executable(hip-compiler main.cpp)

target_link_libraries(hip-compiler PRIVATE
  hip-compiler-lib
)
