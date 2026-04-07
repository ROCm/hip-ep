##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Runtime unit tests - compile mock runtime as a regular static library
# and test wrap_*() functions with GTest.

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# Build mock runtime as a normal static library (not bitcode)
add_library(HipDNNEPRuntimeMock STATIC
  ${CMAKE_SOURCE_DIR}/lib/Runtime/hipdnn_ep_runtime_state.cpp
  ${CMAKE_SOURCE_DIR}/lib/Runtime/hipdnn_ep_runtime_tensor.cpp
  ${CMAKE_SOURCE_DIR}/lib/Runtime/mock/mock_gpu.cpp
  ${CMAKE_SOURCE_DIR}/lib/Runtime/mock/memory.cpp
)

# Include paths - mock must come first so runtime_types.h resolves to mock version
target_include_directories(HipDNNEPRuntimeMock PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/Runtime/mock
  ${CMAKE_SOURCE_DIR}/lib/Runtime
  ${CMAKE_SOURCE_DIR}/include
)

# Link against HipSchemasLib for FlatBuffers generated headers
target_link_libraries(HipDNNEPRuntimeMock PRIVATE HipSchemasLib)

# Depend on schema generation
add_dependencies(HipDNNEPRuntimeMock HipSchemas)

add_executable(runtime-unit-tests
  test_state.cpp
  test_conv.cpp
  test_matmul.cpp
  test_elementwise.cpp
  test_activation.cpp
  test_gather.cpp
  test_cast.cpp
  test_reduce_sum.cpp
  test_norm.cpp
  test_rope.cpp
  test_gqa.cpp
  test_matmul_nbits.cpp
  test_qmoe.cpp
)

target_include_directories(runtime-unit-tests PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/Runtime/mock
  ${CMAKE_SOURCE_DIR}/lib/Runtime
)

target_link_libraries(runtime-unit-tests PRIVATE
  HipDNNEPRuntimeMock
  GTest::gtest
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(runtime-unit-tests)
