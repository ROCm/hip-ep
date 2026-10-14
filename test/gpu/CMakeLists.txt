##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Configuring GPU runtime tests")

add_test(NAME GemvF32TransposedBGpuTest COMMAND test-gemv-f32-transposed-b)
set_tests_properties(GemvF32TransposedBGpuTest PROPERTIES
    LABELS "gpu"
    SKIP_RETURN_CODE 77
)
set_target_properties(test-gemv-f32-transposed-b PROPERTIES
    FOLDER "runtime-tests"
)
