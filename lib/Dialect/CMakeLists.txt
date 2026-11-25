##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_subdirectory(IR)
add_subdirectory(Transforms)
add_subdirectory(Hipsr)
add_subdirectory(Onnx)
if(HIPDNN_EP_BUILD_GGML)
  add_subdirectory(Ggml)
endif()
