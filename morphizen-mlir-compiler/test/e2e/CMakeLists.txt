##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#=============================================================================
# E2E Integration Tests
#=============================================================================
# These tests compile MLIR → DLL and execute them to verify end-to-end
# functionality. All outputs go to CMAKE_CURRENT_BINARY_DIR (build directory).

message(STATUS "Configuring E2E integration tests (CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH})")

# Test 1: Compile demo_two_layer_conv.mlir → demo_two_layer.dll
add_test(NAME CompileDemoConvDLL
    COMMAND mlir-hip-compiler
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/hip-opt/demos/demo_two_layer_conv.mlir"
        --from-onnx-mlir
        -o "${CMAKE_CURRENT_BINARY_DIR}/demo_two_layer.dll"
        --mode dll
        --keep
        -v
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Test 2: Load and test demo_two_layer.dll
add_test(NAME TestDemoConvDLL
    COMMAND test-model-dll "${CMAKE_CURRENT_BINARY_DIR}/demo_two_layer.dll" --verbose --validate
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# TestDemoConvDLL depends on CompileDemoConvDLL
set_tests_properties(TestDemoConvDLL PROPERTIES
    DEPENDS CompileDemoConvDLL
)

#=============================================================================
# ORT Integration Test (optional, requires ONNX Runtime + GTest)
#=============================================================================

find_package(onnxruntime QUIET)

if(TARGET onnxruntime::onnxruntime)
    find_package(GTest QUIET)

    if(TARGET GTest::gtest)
        message(STATUS "Building ORT integration test")

        add_executable(ort_integration_test test_ort_integration.cpp)
        target_link_libraries(ort_integration_test PRIVATE
            onnxruntime::onnxruntime
            GTest::gtest
            GTest::gtest_main
        )

        if(WIN32)
            target_compile_definitions(ort_integration_test PRIVATE "_CRT_SECURE_NO_WARNINGS")
        endif()

        add_test(NAME OrtIntegration COMMAND ort_integration_test)

        set_target_properties(ort_integration_test PROPERTIES FOLDER "tests")
    endif()
else()
    message(STATUS "ONNX Runtime not found, skipping ORT integration test")
endif()
