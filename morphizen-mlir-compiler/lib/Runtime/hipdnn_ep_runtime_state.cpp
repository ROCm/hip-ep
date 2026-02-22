/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Internal runtime state structure
struct RuntimeState {
  hipStream_t stream;
  miopenHandle_t miopen_handle;
  hipblasLtHandle_t hipblas_handle;

  // Array of GPU pointers for constants (size known at compile time)
  void** gpu_constants;
  size_t num_constants;

  // Memory pooling support
  void* pool_base;        // Single large memory pool
  size_t pool_size;       // Total pool size in bytes
  size_t* buffer_offsets; // Offset for each buffer in the pool
  size_t num_buffers;     // Number of buffers in the pool
};

// Runtime state management implementation

int hipdnn_ep_state_init(RuntimeState** out_state,
                         const ConstantRegistry* registry) {
  if (!out_state) {
    fprintf(stderr, "Invalid output parameter to hipdnn_ep_state_init\n");
    return 1;
  }

  // Allocate context struct
  RuntimeState* state = (RuntimeState*)malloc(sizeof(RuntimeState));
  if (!state) {
    fprintf(stderr, "Failed to allocate runtime state\n");
    return 1; // Allocation failed
  }

  // Initialize all fields to null for safe cleanup
  state->stream = nullptr;
  state->miopen_handle = nullptr;
  state->hipblas_handle = nullptr;
  state->gpu_constants = nullptr;
  state->num_constants = registry ? registry->count : 0;
  state->pool_base = nullptr;
  state->pool_size = 0;
  state->buffer_offsets = nullptr;
  state->num_buffers = 0;

  // Allocate constants array (initialized to NULL)
  if (state->num_constants > 0) {
    state->gpu_constants = (void**)calloc(state->num_constants, sizeof(void*));
    if (!state->gpu_constants) {
      fprintf(stderr, "Failed to allocate constants array\n");
      free(state);
      return 1; // Allocation failed
    }
  }

  // Explicitly initialize HIP device before any other operations
  // This ensures device 0 is active and context is properly initialized
  // Critical for DLL execution context where HIP may not auto-initialize
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    fprintf(stderr, "Failed to get HIP device count or no devices available\n");
    free(state->gpu_constants);
    free(state);
    return 2; // Device detection failed
  }

  if (hipSetDevice(0) != hipSuccess) {
    fprintf(stderr, "Failed to set HIP device 0\n");
    free(state->gpu_constants);
    free(state);
    return 3; // Device selection failed
  }

  // Verify device properties are accessible
  // NOTE: Removed dummy hipMalloc/hipFree - let hipStreamCreate initialize
  // context naturally The dummy allocation was causing rocBLAS device
  // enumeration to fail
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    fprintf(stderr, "Failed to get device properties for device 0\n");
    free(state->gpu_constants);
    free(state);
    return 4; // Device properties query failed
  }
  fprintf(stderr, "Device detected: %s (arch=%s)\n", prop.name,
          prop.gcnArchName);

  // WORKAROUND: TheRock SDK Windows bug - gcnArchName is empty due to PAL
  // backend Root cause: Windows uses PAL (Platform Abstraction Layer) instead
  // of HSA (Linux) PAL backend's isa.targetId() query returns empty string in
  // DLL context This causes rocBLAS to fail loading Tensile libraries
  // (arch-specific kernels)
  if (prop.gcnArchName[0] == '\0') {
    fprintf(stderr, "\n=== CRITICAL: TheRock SDK Windows Bug Detected ===\n");
    fprintf(stderr, "gcnArchName is EMPTY (should be 'gfx1100' for W7900)\n");
    fprintf(stderr, "Root cause: PAL backend ISA query fails in DLL context\n");
    fprintf(stderr, "Impact: rocBLAS/hipBLASLt cannot load arch-specific "
                    "Tensile kernels\n\n");

    // Check HSA_OVERRIDE_GFX_VERSION (doesn't work, but check anyway)
    const char* gfx_override = getenv("HSA_OVERRIDE_GFX_VERSION");
    if (gfx_override && gfx_override[0] != '\0') {
      fprintf(stderr,
              "HSA_OVERRIDE_GFX_VERSION=%s (set, but may not work in DLL "
              "context)\n",
              gfx_override);
    } else {
      fprintf(stderr, "HSA_OVERRIDE_GFX_VERSION not set\n");
      fprintf(stderr, "Recommended: export HSA_OVERRIDE_GFX_VERSION=11.0.0\n");
      fprintf(stderr,
              "(Note: This workaround may not fix DLL context issue)\n");
    }

    fprintf(stderr,
            "\nKnown limitation: rocBLAS initialization will likely crash\n");
    fprintf(stderr,
            "Reason: rocBLAS cannot find TensileLibrary_<empty>.dat file\n");
    fprintf(stderr,
            "Workaround: Use standalone .exe instead of .dll execution\n");
    fprintf(stderr, "Long-term fix: Report to ROCm/TheRock GitHub issues\n");
    fprintf(stderr, "===================================================\n\n");
  } else {
    fprintf(stderr,
            "gcnArchName properly populated: '%s' (initialization should "
            "succeed)\n",
            prop.gcnArchName);
  }

  // Create HIP stream
  if (hipStreamCreate(&state->stream) != hipSuccess) {
    fprintf(stderr, "Failed to create HIP stream\n");
    free(state->gpu_constants);
    free(state);
    return 6; // Stream creation failed
  }

  // Create MIOpen handle
  if (miopenCreate(&state->miopen_handle) != miopenStatusSuccess) {
    fprintf(stderr, "Failed to create MIOpen handle\n");
    hipStreamDestroy(state->stream);
    free(state->gpu_constants);
    free(state);
    return 7; // MIOpen creation failed
  }

  // Set stream for MIOpen handle
  if (miopenSetStream(state->miopen_handle, state->stream) !=
      miopenStatusSuccess) {
    fprintf(stderr, "Failed to set MIOpen stream\n");
    miopenDestroy(state->miopen_handle);
    hipStreamDestroy(state->stream);
    free(state->gpu_constants);
    free(state);
    return 8; // Set stream failed
  }

  // DIAGNOSTIC: Check if gcnArchName is still valid AFTER MIOpen initialization
  hipDeviceProp_t prop_after_miopen;
  if (hipGetDeviceProperties(&prop_after_miopen, 0) == hipSuccess) {
    fprintf(stderr,
            "After MIOpen init - gcnArchName='%s' (checking for corruption)\n",
            prop_after_miopen.gcnArchName);
    if (prop_after_miopen.gcnArchName[0] == '\0') {
      fprintf(
          stderr,
          "WARNING: gcnArchName became EMPTY after MIOpen initialization!\n");
      fprintf(stderr,
              "This will cause rocBLAS to fail with 'No devices found'\n");
    }
  }

  // Create hipBLASLt handle
  if (hipblasLtCreate(&state->hipblas_handle) != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Failed to create hipBLASLt handle\n");
    miopenDestroy(state->miopen_handle);
    hipStreamDestroy(state->stream);
    free(state->gpu_constants);
    free(state);
    return 9; // hipBLAS creation failed
  }

  // Upload all constants to GPU using metadata from registry
  if (registry && registry->count > 0) {
    for (size_t i = 0; i < registry->count; i++) {
      const ConstantInfo* info = &registry->constants[i];

      // Allocate GPU memory for this constant
      void* gpu_ptr = nullptr;
      if (hipMalloc(&gpu_ptr, info->size_bytes) != hipSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory for constant %zu\n", i);
        // Cleanup already uploaded constants
        for (size_t j = 0; j < i; j++) {
          if (state->gpu_constants[j]) {
            hipFree(state->gpu_constants[j]);
          }
        }
        hipblasLtDestroy(state->hipblas_handle);
        miopenDestroy(state->miopen_handle);
        hipStreamDestroy(state->stream);
        free(state->gpu_constants);
        free(state);
        return 9; // Constant upload failed
      }

      // Copy constant data from CPU to GPU
      if (hipMemcpy(gpu_ptr, info->cpu_data, info->size_bytes,
                    hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "Failed to copy constant %zu to GPU\n", i);
        hipFree(gpu_ptr);
        // Cleanup already uploaded constants
        for (size_t j = 0; j < i; j++) {
          if (state->gpu_constants[j]) {
            hipFree(state->gpu_constants[j]);
          }
        }
        hipblasLtDestroy(state->hipblas_handle);
        miopenDestroy(state->miopen_handle);
        hipStreamDestroy(state->stream);
        free(state->gpu_constants);
        free(state);
        return 9; // Constant upload failed
      }

      // Store GPU pointer in state
      state->gpu_constants[i] = gpu_ptr;
    }
  }

  // Success - return initialized state
  *out_state = state;
  return 0;
}

int hipdnn_ep_state_cleanup(RuntimeState* state) {
  if (!state) {
    fprintf(stderr, "Invalid runtime state in cleanup\n");
    return 0; // Best-effort - don't fail
  }

  // Best-effort cleanup - continue even if operations fail
  // Cleanup in reverse order of initialization (LIFO)

  // Synchronize stream to ensure all GPU operations complete
  if (state->stream) {
    hipStreamSynchronize(state->stream);
  }

  // Free memory pool (if allocated)
  if (state->pool_base) {
    hipFree(state->pool_base);
  }
  if (state->buffer_offsets) {
    free(state->buffer_offsets);
  }

  // Free all constants (best-effort)
  if (state->gpu_constants) {
    for (size_t i = 0; i < state->num_constants; i++) {
      if (state->gpu_constants[i]) {
        hipFree(state->gpu_constants[i]);
      }
    }
    free(state->gpu_constants);
  }

  // Destroy hipBLASLt handle
  if (state->hipblas_handle) {
    hipblasLtDestroy(state->hipblas_handle);
  }

  // Destroy MIOpen handle
  if (state->miopen_handle) {
    miopenDestroy(state->miopen_handle);
  }

  // Destroy HIP stream
  if (state->stream) {
    hipStreamDestroy(state->stream);
  }

  // Free the context struct itself
  free(state);

  return 0; // Best-effort cleanup always returns success
}

void* hipdnn_ep_state_get_stream(RuntimeState* state) {
  return state ? static_cast<void*>(state->stream) : nullptr;
}

void* hipdnn_ep_state_get_miopen_handle(RuntimeState* state) {
  return state ? static_cast<void*>(state->miopen_handle) : nullptr;
}

//==============================================================================
// Backward Compatibility Wrappers
//==============================================================================

extern "C" {

// Legacy: runtime_state_init
int runtime_state_init(void** out_state) {
  return hipdnn_ep_state_init(reinterpret_cast<RuntimeState**>(out_state),
                              nullptr);
}

// Legacy: runtime_state_cleanup
int runtime_state_cleanup(void* state) {
  return hipdnn_ep_state_cleanup(static_cast<RuntimeState*>(state));
}

// Legacy: runtime_prepare_inference
int runtime_prepare_inference(void* state, void* inputs_ptr, void* outputs_ptr,
                              void** out_data) {
  *out_data = malloc(8); // Dummy allocation
  return 0;
}

// Legacy: runtime_cleanup_inference
int runtime_cleanup_inference(void* state, void* data, void* outputs_ptr) {
  if (data) {
    free(data);
  }
  return 0;
}

} // extern "C"

//==============================================================================
// Memory Pooling Support
//==============================================================================

extern "C" {

int hipdnn_ep_pool_init(RuntimeState* state, size_t pool_size,
                        const size_t* buffer_offsets, size_t num_buffers) {
  if (!state) {
    fprintf(stderr, "Invalid state parameter to hipdnn_ep_pool_init\n");
    return 1;
  }

  // Allocate the memory pool
  if (pool_size > 0) {
    if (hipMalloc(&state->pool_base, pool_size) != hipSuccess) {
      fprintf(stderr, "Failed to allocate memory pool of size %zu bytes\n",
              pool_size);
      return 2; // Pool allocation failed
    }
  } else {
    state->pool_base = nullptr;
  }

  // Store pool metadata
  state->pool_size = pool_size;
  state->num_buffers = num_buffers;

  // Copy buffer offsets array
  if (num_buffers > 0 && buffer_offsets) {
    state->buffer_offsets = (size_t*)malloc(sizeof(size_t) * num_buffers);
    if (!state->buffer_offsets) {
      fprintf(stderr, "Failed to allocate buffer offsets array\n");
      if (state->pool_base) {
        hipFree(state->pool_base);
        state->pool_base = nullptr;
      }
      return 1; // Allocation failed
    }
    memcpy(state->buffer_offsets, buffer_offsets, sizeof(size_t) * num_buffers);
  } else {
    state->buffer_offsets = nullptr;
  }

  return 0; // Success
}

void* hipdnn_ep_get_buffer_from_pool(RuntimeState* state, size_t index) {
  if (!state || !state->pool_base) {
    fprintf(stderr, "Invalid state or pool not initialized\n");
    return nullptr;
  }

  if (index >= state->num_buffers) {
    fprintf(stderr, "Buffer index %zu out of range (num_buffers = %zu)\n",
            index, state->num_buffers);
    return nullptr;
  }

  // Return pointer at pool_base + offset
  char* pool_ptr = static_cast<char*>(state->pool_base);
  size_t offset = state->buffer_offsets[index];
  return pool_ptr + offset;
}

} // extern "C"
