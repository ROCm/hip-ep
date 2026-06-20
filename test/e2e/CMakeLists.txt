##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

    # Per-test isolated directory prevents constants file collisions under
    # parallel ctest. Each test writes its own constants.bin here.
    set(test_dir "${CMAKE_CURRENT_BINARY_DIR}/${test_name}")
    file(MAKE_DIRECTORY "${test_dir}")

    # Output DLL path (inside per-test dir)
    set(dll_output "${test_dir}/${test_name}.dll")

    # Test 1: Compile .mlir → .dll
    # WORKING_DIRECTORY is set to the per-test dir so constants.bin
    # (the default constants filename) is written there.
    # --mode NATIVE: the default artifact format is now LLVM_IR (.bc), so
    # request a native .dll explicitly to keep the ELF init/fini check below
    # meaningful and exercise the native loader path.
    add_test(NAME E2E_Compile_${test_name}
        COMMAND $<TARGET_FILE:hip-compiler>
            "${test_file}"
            -o "${dll_output}"
            --mode NATIVE
        WORKING_DIRECTORY "${test_dir}"
    )

    # Test 2: Execute .dll
    # WORKING_DIRECTORY matches the compile step so hip-test resolves
    # constants.bin from the same per-test dir via DiskFileSystem(".").
    add_test(NAME E2E_Execute_${test_name}
        COMMAND $<TARGET_FILE:hip-test>
            "${dll_output}"
            --verbose
            --validate
        WORKING_DIRECTORY "${test_dir}"
    )

    set_tests_properties(E2E_Execute_${test_name} PROPERTIES
        DEPENDS E2E_Compile_${test_name}
    )
endforeach()

list(LENGTH MLIR_E2E_TESTS num_tests)
message(STATUS "Configured ${num_tests} E2E test(s)")

# Linux-only build-stage regression check for two ELF-init bugs that
# actually shipped during the Linux port:
#   1. DT_INIT_ARRAY missing  -> static ctors silently never run -> SIGFPE
#      on first emplace into a zero-init `static std::unordered_map`.
#   2. DT_FINI_ARRAY missing  -> dlclose can't drain __cxa_finalize ->
#      SIGSEGV at process exit walking stale dtor pointers.
# Plus a no-`.ctors` check: a populated .ctors section means LLVM
# regressed back to UseInitArray=false (bug 1 above).
if(NOT WIN32)
    set(_init_fini_check_dll
        "${CMAKE_CURRENT_BINARY_DIR}/test_add_model/test_add_model.dll")
    add_test(NAME E2E_DLL_HasInitFiniArrays
        COMMAND ${CMAKE_COMMAND} -E env bash -c
            "readelf -d '${_init_fini_check_dll}' | grep -q '(INIT_ARRAY)' && \
             readelf -d '${_init_fini_check_dll}' | grep -q '(FINI_ARRAY)' && \
             ! readelf -S '${_init_fini_check_dll}' | grep -q '\\.ctors '"
    )
    set_tests_properties(E2E_DLL_HasInitFiniArrays PROPERTIES
        DEPENDS E2E_Compile_test_add_model
    )
endif()
