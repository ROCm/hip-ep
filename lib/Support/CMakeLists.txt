##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipSupport STATIC CrashHandler.cpp)

target_include_directories(HipSupport PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(HipSupport PUBLIC cpptrace::cpptrace)

# Shared constants.bin file I/O. Its own target (not folded into any
# conversion) so producers -- OnnxToHip, the MorphiZen EP, hipsr-externalize-
# constants -- depend on this rather than on each other's pipeline libs.
add_library(HipConstantsIO STATIC ConstantsIO.cpp)

target_include_directories(HipConstantsIO PRIVATE
  ${CMAKE_SOURCE_DIR}/morphizen/morphizen-foundation/include  # file_io.hpp
)
