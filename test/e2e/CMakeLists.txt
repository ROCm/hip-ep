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
    add_test(NAME E2E_Compile_${test_name}
        COMMAND $<TARGET_FILE:hip-compiler>
            "${test_file}"
            -o "${dll_output}"
        WORKING_DIRECTORY "${test_dir}"
    )

    # Test 2: Execute .dll
    # WORKING_DIRECTORY matches the compile step so hip-test-dll resolves
    # constants.bin from the same per-test dir via DiskFileSystem(".").
    add_test(NAME E2E_Execute_${test_name}
        COMMAND $<TARGET_FILE:hip-test-dll>
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

# Linux-only: verify the per-model DLL has both DT_INIT_ARRAY and DT_FINI_ARRAY
# populated. Without DT_INIT_ARRAY, LLVM's default UseInitArray=false routes
# C++ static ctors into legacy .ctors which glibc's loader drops, leaving
# every `static std::unordered_map` etc. at zero-init BSS (the first emplace
# does `hash % 0` -> SIGFPE inside the model). Without DT_FINI_ARRAY (because
# crtbeginS.o / crtendS.o weren't linked into the DLL), dlclose never calls
# __cxa_finalize(&__dso_handle), so static dtors stay in libc's global atexit
# list and crash at process exit() with stale function pointers.
#
# Both regressions actually shipped during the Linux port (28/28 SIGFPE in
# inference_compute, and SIGSEGV in libc exit() after `OK - X output
# tensor(s)`); this test catches them at the build stage, before they reach
# a runtime crash.
if(NOT WIN32)
    # Three checks on the per-model DLL:
    #   1. DT_INIT_ARRAY present (dynamic loader hook for ctors exists)
    #   2. DT_FINI_ARRAY present (dlclose hook for dtors exists)
    #   3. NO `.ctors` section (legacy section; if present, LLVM codegen
    #      routed C++ static ctors there instead of .init_array — the
    #      UseInitArray=false default that caused 28/28 SIGFPE)
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
