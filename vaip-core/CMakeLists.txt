##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_onnxruntime.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/proto.cmake)
# MorphiZen must be build along with onnxruntime source tree
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/morphizen-core-static.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hash-library.cmake)
if(morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/morphizen-core-dynamic.cmake)
endif(morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/install.cmake)
