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

# Compiles only the pool/host-scratch site portion of runtime_state.cpp against
# tiny mock HIP allocation functions. This directly proves caller/helper local
# zero identities map to distinct pointers and child growth leaves parents live.
add_executable(test-pool-sites
    test_pool_sites.cpp
    ${CMAKE_SOURCE_DIR}/lib/Runtime/hipdnn_ep_runtime_state.cpp
)

target_compile_features(test-pool-sites PRIVATE cxx_std_17)

target_include_directories(test-pool-sites PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Runtime
    ${CMAKE_SOURCE_DIR}/lib/Runtime/mock
)

target_compile_definitions(test-pool-sites PRIVATE
    HIPDNN_EP_POOL_UNIT_TEST
    HIPDNN_EP_RT_NO_EXPORT
)

add_test(NAME PoolSitesUnitTest COMMAND test-pool-sites)

set_target_properties(test-pool-sites PROPERTIES FOLDER "runtime-tests")

# Exercises the void CustomOp ABI's exception-to-OrtStatus boundary with a
# generated-compute failure seam. The fake OrtApi needs only CreateStatus, so
# this test executes without loading HIP or an ORT session.
add_executable(test-custom-op-compute-status
    test_custom_op_compute_status.cpp
)

target_compile_features(test-custom-op-compute-status PRIVATE cxx_std_17)

target_include_directories(test-custom-op-compute-status PRIVATE
    ${CMAKE_SOURCE_DIR}/morphizen/ort-bridge/src
)

target_link_libraries(test-custom-op-compute-status PRIVATE
    onnxruntime::onnxruntime
)

add_test(NAME CustomOpComputeStatusUnitTest
    COMMAND test-custom-op-compute-status)

set_target_properties(test-custom-op-compute-status
    PROPERTIES FOLDER "runtime-tests")

# Pins the opaque TensorBuffer storage/initialization ABI used by generated
# inference cleanup. This source has no HIP dependency.
add_executable(test-tensor-buffer-lifecycle
    test_tensor_buffer_lifecycle.cpp
    ${CMAKE_SOURCE_DIR}/lib/Runtime/tensor_buffer.cpp
)

target_compile_features(test-tensor-buffer-lifecycle PRIVATE cxx_std_17)

target_include_directories(test-tensor-buffer-lifecycle PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Runtime
)

add_test(NAME TensorBufferLifecycleUnitTest
    COMMAND test-tensor-buffer-lifecycle)

set_target_properties(test-tensor-buffer-lifecycle
    PROPERTIES FOLDER "runtime-tests")

# Validates current/stale/missing/malformed artifact ABI tokens, strict
# EPContext metadata parsing, and both LLVM_IR/NATIVE loader-selection seams.
# The test calls only pure metadata/symbol-validation helpers and needs no GPU.
add_executable(test-artifact-abi test_artifact_abi.cpp)
target_compile_features(test-artifact-abi PRIVATE cxx_std_17)
target_include_directories(test-artifact-abi PRIVATE
    ${CMAKE_SOURCE_DIR}/backend-mlir-compiler/custom-op-mlir/src
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test-artifact-abi PRIVATE
    mlir_compilation_proto
)
add_test(NAME ArtifactAbiUnitTest COMMAND test-artifact-abi)
set_target_properties(test-artifact-abi PROPERTIES FOLDER "runtime-tests")

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

add_executable(test-cache-identity test_cache_identity.cpp)
target_compile_features(test-cache-identity PRIVATE cxx_std_17)
target_link_libraries(test-cache-identity PRIVATE
    morphizen-core-static
)
add_test(NAME CacheIdentityUnitTest COMMAND test-cache-identity)
set_target_properties(test-cache-identity PROPERTIES FOLDER "runtime-tests")

# Shared MatMul/Gemm runtime preflight and mock ABI behavior. Dispatch counters
# prove invalid contractions return before the simulated BLAS boundary.
add_executable(test-matmul-gemm-contract
    test_matmul_gemm_contract.cpp
    ${CMAKE_SOURCE_DIR}/lib/Runtime/mock/mock_gpu.cpp
)
target_compile_features(test-matmul-gemm-contract PRIVATE cxx_std_17)
target_include_directories(test-matmul-gemm-contract PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Runtime
    ${CMAKE_SOURCE_DIR}/lib/Runtime/mock
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_definitions(test-matmul-gemm-contract PRIVATE
    HIPDNN_EP_RT_NO_EXPORT
    HIPDNN_EP_RUNTIME_TESTING
)
add_test(NAME MatmulGemmContractUnitTest COMMAND test-matmul-gemm-contract)
set_target_properties(test-matmul-gemm-contract PROPERTIES FOLDER "runtime-tests")
