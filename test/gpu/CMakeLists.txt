##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Configuring GPU runtime tests")

add_test(NAME GemvBf16F32TransposedBGpuTest
    COMMAND test-gemv-bf16-f32-transposed-b)
set_tests_properties(GemvBf16F32TransposedBGpuTest PROPERTIES
    LABELS "gpu"
    SKIP_RETURN_CODE 77
)
set_target_properties(test-gemv-bf16-f32-transposed-b PROPERTIES
    FOLDER "runtime-tests"
)

add_test(NAME MatmulTypedGpuTest COMMAND test-matmul-typed)
set_tests_properties(MatmulTypedGpuTest PROPERTIES
    LABELS "gpu"
    SKIP_RETURN_CODE 77
)
set_target_properties(test-matmul-typed PROPERTIES FOLDER "runtime-tests")
