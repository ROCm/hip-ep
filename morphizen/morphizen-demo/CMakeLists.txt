##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# morphizen-demo: Demo projects for MorphiZen
# This is integrated as a subdirectory of MorphiZen, not a standalone project.

add_subdirectory(proto)
add_subdirectory(level-1-pass-relu-dq)
add_subdirectory(custom-op-relu-dq)
# test_onnx_runner requires Boost::program_options
if(morphizen_ENABLE_BOOST)
  add_subdirectory(test)
endif()
