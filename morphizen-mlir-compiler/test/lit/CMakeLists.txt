##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# LIT tests for morphizen-mlir-compiler
# Simplified - tests are run via llvm-lit directly (no site config needed)

# Add test target
add_custom_target(check-morphizen-mlir-lit
    COMMAND ${Python3_EXECUTABLE} -m lit -v ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS hip-opt mlir-hip-compiler
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running LIT tests for morphizen-mlir-compiler"
)

# Add to CTest
add_test(
    NAME MorphizenMLIRLitTests
    COMMAND ${Python3_EXECUTABLE} -m lit -v ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
