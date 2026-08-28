##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#=============================================================================
# Runtime unit tests (GPU-free)
#=============================================================================
# Compiles individual runtime translation units natively against the MOCK
# runtime types and exercises them without a GPU. Registered with CTest.

message(STATUS "Configuring runtime unit tests")

# Compiles output_allocator.cpp natively (the same source that also becomes
# runtime bitcode) so the allocator contract is tested directly.
add_executable(test-output-allocator
    test_output_allocator.cpp
    ${CMAKE_SOURCE_DIR}/lib/Runtime/output_allocator.cpp
)

target_compile_features(test-output-allocator PRIVATE cxx_std_17)

# lib/Runtime  -> hipdnn_ep_runtime.h, runtime_state_internal.h, hipdnn_ep_errors.h
# lib/Runtime/mock -> runtime_types.h resolves to the GPU-free mock typedefs
#   (void* handles), so nothing here needs the HIP/MIOpen/hipBLASLt SDK.
target_include_directories(test-output-allocator PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Runtime
    ${CMAKE_SOURCE_DIR}/lib/Runtime/mock
)

# Drop __declspec(dllexport) on the setter for this native build: the test does
# not need to export it, and the exported decl/def is only required for the
# bitcode path. Without this, MSVC errors C2375 on the decl/def linkage mismatch.
target_compile_definitions(test-output-allocator PRIVATE HIPDNN_EP_RT_NO_EXPORT)

add_test(NAME OutputAllocatorUnitTest COMMAND test-output-allocator)

set_target_properties(test-output-allocator PROPERTIES FOLDER "runtime-tests")
