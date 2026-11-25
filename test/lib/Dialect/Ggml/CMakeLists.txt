##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ggml-analyses: the ggml dialect's analyses printed for the lit tests
# (test/lit/Dialect/Ggml).
add_executable(ggml-analyses ggml-analyses.cpp)
target_link_libraries(ggml-analyses PRIVATE GgmlTransforms)
