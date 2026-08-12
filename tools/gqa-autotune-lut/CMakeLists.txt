##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Generates GQA autotune LUT rows from the cost model, for geometries nobody
# measured. Offline only: the runtime resolves configs purely from the LUT, so
# the model is never linked into the EP or the kernel library.
#
# Host C++ with no HIP and no flatbuffers dependency -- it emits JSON that flatc
# turns into the shipped binary (see etc/gqa_autotune/README.md).
add_executable(gqa-lut-rows-from-model
    gqa_lut_rows_from_model.cpp
    ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/hip/autotune/gqa/gqa_cost_model.cpp
)

target_compile_features(gqa-lut-rows-from-model PRIVATE cxx_std_17)
target_include_directories(gqa-lut-rows-from-model PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/hip/autotune/gqa
)

set_target_properties(gqa-lut-rows-from-model PROPERTIES FOLDER "tools")
