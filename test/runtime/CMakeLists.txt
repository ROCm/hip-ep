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
#   (void* handles), so nothing here needs the HIP/hipBLASLt SDK.
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

# GQA offline policy tests are GPU-free: compatibility fields use wildcard
# arch/ROCm values, while exact -> bucket -> heuristic resolution and the
# FlatBuffer loader exercise the same implementation compiled into runtime.bc.
add_executable(test-gqa-autotune
    test_gqa_autotune.cpp
    ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/hip/autotune/gqa/gqa_autotune.cpp
)

target_compile_features(test-gqa-autotune PRIVATE cxx_std_17)
target_compile_definitions(test-gqa-autotune PRIVATE
    HIPDNN_EP_GQA_AUTOTUNE_GPU_FREE=1
)
target_include_directories(test-gqa-autotune PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Runtime
    ${CMAKE_SOURCE_DIR}/lib/Runtime/real
    ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/hip/autotune/gqa
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/morphizen/morphizen-foundation/include
)
target_link_libraries(test-gqa-autotune PRIVATE HipSchemasLib)

add_test(NAME GqaAutotuneUnitTest COMMAND test-gqa-autotune)
set_target_properties(test-gqa-autotune PROPERTIES FOLDER "runtime-tests")
add_executable(test-symbolic-dims test_symbolic_dims.cpp)
target_compile_features(test-symbolic-dims PRIVATE cxx_std_17)
target_link_libraries(test-symbolic-dims PRIVATE morphizen-graph)
add_test(NAME SymbolicDimsUnitTest COMMAND test-symbolic-dims)
set_target_properties(test-symbolic-dims PROPERTIES FOLDER "runtime-tests")

add_executable(test-symbolic-dims-mlir test_symbolic_dims_mlir.cpp)
target_compile_features(test-symbolic-dims-mlir PRIVATE cxx_std_17)
target_include_directories(test-symbolic-dims-mlir PRIVATE
    ${CMAKE_SOURCE_DIR}/morphizen/mlir-imp/src
)
target_include_directories(test-symbolic-dims-mlir SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(test-symbolic-dims-mlir PRIVATE
    mlir-imp
    morphizen-graph
)
add_test(NAME SymbolicDimsMLIRUnitTest COMMAND test-symbolic-dims-mlir)
set_target_properties(test-symbolic-dims-mlir PROPERTIES FOLDER "runtime-tests")
