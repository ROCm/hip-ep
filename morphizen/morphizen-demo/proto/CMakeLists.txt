##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Header-only replacement for the former relu_dq.proto library.
add_library(relu_dq_proto INTERFACE)
target_include_directories(relu_dq_proto INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
