##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#=============================================================================
# E2E Integration Tests
#=============================================================================
# These tests compile MLIR → DLL and execute them to verify end-to-end
# functionality. Auto-discovers .mlir files from test/lit/e2e/.

message(STATUS "Configuring E2E integration tests")

# Auto-discover all .mlir test files from test/lit/e2e/
file(GLOB MLIR_E2E_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../lit/e2e/*.mlir")

# For each .mlir file, create compile + execute test pair
foreach(test_file ${MLIR_E2E_TESTS})
    # Extract test name (filename without extension)
    get_filename_component(test_name ${test_file} NAME_WE)

    # Output DLL path
    set(dll_output "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.dll")

    # Test 1: Compile .mlir → .dll
    add_test(NAME E2E_Compile_${test_name}
        COMMAND $<TARGET_FILE:morphizen-compile>
            "${test_file}"
            -o "${dll_output}"
            --mode dll
            -v
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # Test 2: Execute .dll
    add_test(NAME E2E_Execute_${test_name}
        COMMAND $<TARGET_FILE:test-model-dll>
            "${dll_output}"
            --verbose
            --validate
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # Execution depends on compilation
    set_tests_properties(E2E_Execute_${test_name} PROPERTIES
        DEPENDS E2E_Compile_${test_name}
    )
endforeach()

list(LENGTH MLIR_E2E_TESTS num_tests)
message(STATUS "Configured ${num_tests} E2E test(s)")
