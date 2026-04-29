##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipSupport STATIC CrashHandler.cpp)

target_include_directories(HipSupport PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(HipSupport PUBLIC cpptrace::cpptrace)
