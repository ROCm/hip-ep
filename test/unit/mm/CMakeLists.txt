##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Memory Manager unit tests (GoogleTest)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(MM_TEST_SOURCES
    test_hal.cpp
    test_handle_table.cpp
    test_lifecycle.cpp
    test_pool.cpp
    test_arena.cpp
    test_core.cpp
    test_free_list.cpp
    test_kv_block.cpp
    test_kv_fork.cpp
    test_block_table.cpp
    test_kv_cache_state.cpp
)

add_executable(mm_tests ${MM_TEST_SOURCES})

target_link_libraries(mm_tests PRIVATE
    HipDnnEpMemoryManager
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(mm_tests)
