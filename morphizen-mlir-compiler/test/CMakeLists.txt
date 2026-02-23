##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# LIT tests
if(ONNX_HIP_INCLUDE_LIT_TESTS)
    add_subdirectory(lit)
endif()

# E2E tests (includes ORT integration test)
add_subdirectory(e2e)
