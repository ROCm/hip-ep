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
#include <glog/logging.h>

// Windows defines ERROR macro which conflicts with our enum
#ifdef ERROR
#undef ERROR
#endif

namespace rocm_ep {

/**
 * GPU Operation Timeout Result
 */
enum class TimeoutStatus {
  SUCCESS,           // Operation completed successfully
  TIMEOUT,          // Operation timed out
  ERROR             // Error occurred
};

/**
 * Helper function to wait for HIP stream with timeout
 * 
 * @param stream HIP stream to wait for
 * @param timeout_ms Timeout in milliseconds
 * @return TimeoutStatus indicating success, timeout, or error
 */
inline TimeoutStatus WaitStreamWithTimeout(hipStream_t stream, int timeout_ms) {
  const int poll_interval_ms = 10;  // Poll every 10ms
  auto start = std::chrono::steady_clock::now();
  
  while (true) {
    // Query stream status (non-blocking)
    hipError_t err = hipStreamQuery(stream);
    
    if (err == hipSuccess) {
      // All operations completed
      return TimeoutStatus::SUCCESS;
    } else if (err == hipErrorNotReady) {
      // Operations still in progress
      auto elapsed = std::chrono::steady_clock::now() - start;
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
      
      if (elapsed_ms >= timeout_ms) {
        LOG(ERROR) << "[ROCm Timeout] GPU operation timed out after " << elapsed_ms << "ms";
        return TimeoutStatus::TIMEOUT;
      }
      
      // Sleep briefly to avoid busy-waiting
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    } else {
      // Error occurred
      LOG(ERROR) << "[ROCm Timeout] hipStreamQuery failed: " 
                 << hipGetErrorString(err) << " (" << err << ")";
      return TimeoutStatus::ERROR;
    }
  }
}

} // namespace rocm_ep

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

int main(int argc, char** argv) {
    // Initialize glog
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    
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
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "   All tests completed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
