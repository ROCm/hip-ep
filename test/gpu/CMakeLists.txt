##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Configuring GPU runtime tests")

add_test(NAME CastBf16GpuTest COMMAND test-cast-bf16)
set_tests_properties(CastBf16GpuTest PROPERTIES
    LABELS "gpu"
    SKIP_RETURN_CODE 77
)
set_target_properties(test-cast-bf16 PROPERTIES
    FOLDER "runtime-tests"
)
