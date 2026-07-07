##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(CMAKE_CXX_STANDARD 17)
if(MSVC)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_options.msvc.cmake)
else()
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_options.linux.cmake)
endif()
