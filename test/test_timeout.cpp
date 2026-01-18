// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

/**
 * GPU Timeout Test
 * 
 * This test demonstrates the GPU timeout mechanism by intentionally
 * creating scenarios that test timeout behavior.
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>

// Include the timeout handling code
#include "../custom-op-rocm/src/custom_op.hpp"

void test_immediate_completion() {
    std::cout << "\n=== Test 1: Immediate Completion ===" << std::endl;
    
    // Create a stream and immediately query it (should be ready)
    hipStream_t stream;
    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        std::cerr << "Failed to create stream: " << hipGetErrorString(err) << std::endl;
        return;
    }
    
    // Test timeout on empty stream (should succeed immediately)
    auto status = rocm_ep::WaitStreamWithTimeout(stream, 1000);
    
    if (status == rocm_ep::TimeoutStatus::SUCCESS) {
        std::cout << "✅ PASS: Empty stream completed immediately" << std::endl;
    } else {
        std::cout << "❌ FAIL: Empty stream did not complete" << std::endl;
    }
    
    hipStreamDestroy(stream);
}

void test_short_operation() {
    std::cout << "\n=== Test 2: Short Operation ===" << std::endl;
    
    hipStream_t stream;
    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        std::cerr << "Failed to create stream: " << hipGetErrorString(err) << std::endl;
        return;
    }
    
    // Allocate device memory and perform a simple operation
    float* d_data = nullptr;
    size_t size = 1024 * 1024 * sizeof(float);
    err = hipMalloc(&d_data, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        hipStreamDestroy(stream);
        return;
    }
    
    // Launch a simple memset operation
    err = hipMemsetAsync(d_data, 0, size, stream);
    if (err != hipSuccess) {
        std::cerr << "Failed to launch memset: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        hipStreamDestroy(stream);
        return;
    }
    
    // Wait with generous timeout (should complete quickly)
    auto start = std::chrono::steady_clock::now();
    auto status = rocm_ep::WaitStreamWithTimeout(stream, 5000);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (status == rocm_ep::TimeoutStatus::SUCCESS) {
        std::cout << "✅ PASS: Short operation completed in " << elapsed_ms << "ms" << std::endl;
    } else {
        std::cout << "❌ FAIL: Short operation did not complete" << std::endl;
    }
    
    hipFree(d_data);
    hipStreamDestroy(stream);
}

void test_timeout_detection() {
    std::cout << "\n=== Test 3: Timeout Detection ===" << std::endl;
    std::cout << "Note: This test verifies timeout logic, not actual GPU hang" << std::endl;
    
    hipStream_t stream;
    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        std::cerr << "Failed to create stream: " << hipGetErrorString(err) << std::endl;
        return;
    }
    
    // Allocate device memory
    float* d_data = nullptr;
    size_t size = 1024 * 1024 * sizeof(float);
    err = hipMalloc(&d_data, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        hipStreamDestroy(stream);
        return;
    }
    
    // Launch multiple operations to keep the stream busy
    for (int i = 0; i < 100; i++) {
        hipMemsetAsync(d_data, i % 256, size, stream);
    }
    
    // Try to wait with a VERY short timeout (likely to timeout if operations are still running)
    // Note: On fast GPUs, even this might complete in time
    auto start = std::chrono::steady_clock::now();
    auto status = rocm_ep::WaitStreamWithTimeout(stream, 1);  // 1ms timeout
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (status == rocm_ep::TimeoutStatus::TIMEOUT) {
        std::cout << "✅ PASS: Timeout detected after " << elapsed_ms << "ms (as expected with 1ms limit)" << std::endl;
    } else if (status == rocm_ep::TimeoutStatus::SUCCESS) {
        std::cout << "⚠️  INFO: Operations completed faster than 1ms (very fast GPU!)" << std::endl;
        std::cout << "         Elapsed: " << elapsed_ms << "ms" << std::endl;
    } else {
        std::cout << "❌ FAIL: Unexpected error status" << std::endl;
    }
    
    // Now wait with proper timeout to let operations complete
    hipStreamSynchronize(stream);
    
    hipFree(d_data);
    hipStreamDestroy(stream);
}

void test_hip_context_timeout() {
    std::cout << "\n=== Test 4: HipContext Timeout Method ===" << std::endl;
    
    // Get the HIP context instance
    auto& ctx = rocm_ep::HipContext::instance();
    
    if (!ctx.is_initialized()) {
        std::cout << "❌ FAIL: HIP context not initialized (no GPU available?)" << std::endl;
        return;
    }
    
    // Get the stream and launch a simple operation
    hipStream_t stream = ctx.stream();
    float* d_data = nullptr;
    size_t size = 1024 * sizeof(float);
    hipError_t err = hipMalloc(&d_data, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        return;
    }
    
    hipMemsetAsync(d_data, 42, size, stream);
    
    // Test the HipContext timeout method with generous timeout
    auto status = ctx.sync_stream_with_timeout(5000);
    
    if (status == rocm_ep::TimeoutStatus::SUCCESS) {
        std::cout << "✅ PASS: HipContext timeout method works correctly" << std::endl;
    } else {
        std::cout << "❌ FAIL: HipContext timeout method failed" << std::endl;
    }
    
    hipFree(d_data);
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "   GPU Timeout Mechanism Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Check if GPU is available
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        std::cerr << "No AMD GPU detected! Tests cannot run." << std::endl;
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    std::cout << "\nDetected GPU: " << props.name << std::endl;
    std::cout << "GCN Arch: " << props.gcnArchName << std::endl;
    
    // Run tests
    test_immediate_completion();
    test_short_operation();
    test_timeout_detection();
    test_hip_context_timeout();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "   All tests completed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
