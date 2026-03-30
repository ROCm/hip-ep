/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "debug_log.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include "model_metadata_generated.h"
#include "morphizen-foundation/file_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Macro for best-effort cleanup: logs errors but continues cleanup
#define HIP_CLEANUP(expr)                                                      \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      fprintf(stderr, "Warning: " #expr " failed with error %d\n", (int)_err); \
    }                                                                          \
  } while (0)

// Runtime state management implementation

int hipdnn_ep_state_init_with_fs(RuntimeState **out_state, void *fs,
                                 const void *metadata_blob, size_t blob_size) {
  if (!out_state || !fs) {
    fprintf(stderr, "Invalid arguments to hipdnn_ep_state_init_with_fs\n");
    return 1;
  }

  // Allocate context struct
  RuntimeState *state = (RuntimeState *)malloc(sizeof(RuntimeState));
  if (!state) {
    fprintf(stderr, "Failed to allocate runtime state\n");
    return 1;
  }

  // Initialize all fields to null for safe cleanup
  state->stream = nullptr;
  state->miopen_handle = nullptr;
  state->hipblas_handle = nullptr;
  state->gpu_constants_blob = nullptr;
  state->constants_blob_is_host = false;
  state->gpu_constants = nullptr;
  state->num_constants = 0;
  state->pool_base = nullptr;
  state->pool_size = 0;
  state->buffer_offsets = nullptr;
  state->num_buffers = 0;
  state->workspace = nullptr;
  state->workspace_size = 0;
  state->hipdnn_handle = nullptr;
  state->hipdnn_graph_registry = nullptr;

  // Explicitly initialize HIP device before any other operations
  // This ensures device 0 is active and context is properly initialized
  // Critical for DLL execution context where HIP may not auto-initialize
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    fprintf(stderr, "Failed to get HIP device count or no devices available\n");
    free(state);
    return 2;
  }

  if (hipSetDevice(0) != hipSuccess) {
    fprintf(stderr, "Failed to set HIP device 0\n");
    free(state);
    return 3;
  }

  // Verify device properties are accessible
  // NOTE: Removed dummy hipMalloc/hipFree - let hipStreamCreate initialize
  // context naturally. The dummy allocation was causing rocBLAS device
  // enumeration to fail.
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    fprintf(stderr, "Failed to get device properties for device 0\n");
    free(state);
    return 4;
  }
  RUNTIME_DEBUG_LOG("Device detected: %s (arch=%s)\n", prop.name,
                    prop.gcnArchName);

  // WORKAROUND: TheRock SDK Windows bug - gcnArchName is empty due to PAL
  // backend. Root cause: Windows uses PAL (Platform Abstraction Layer) instead
  // of HSA (Linux). PAL backend's isa.targetId() query returns empty string in
  // DLL context. This causes rocBLAS to fail loading Tensile libraries
  // (arch-specific kernels).
  if (prop.gcnArchName[0] == '\0') {
    fprintf(stderr, "\n=== CRITICAL: TheRock SDK Windows Bug Detected ===\n");
    fprintf(stderr, "gcnArchName is EMPTY (should be 'gfx1100' for W7900)\n");
    fprintf(stderr, "Root cause: PAL backend ISA query fails in DLL context\n");
    fprintf(stderr, "Impact: rocBLAS/hipBLASLt cannot load arch-specific "
                    "Tensile kernels\n\n");

    const char *gfx_override = getenv("HSA_OVERRIDE_GFX_VERSION");
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
    RUNTIME_DEBUG_LOG(
        "gcnArchName properly populated: '%s' (initialization should "
        "succeed)\n",
        prop.gcnArchName);
  }

  // Create HIP stream
  if (hipStreamCreate(&state->stream) != hipSuccess) {
    fprintf(stderr, "Failed to create HIP stream\n");
    free(state);
    return 6;
  }

  // Create MIOpen handle
  if (miopenCreate(&state->miopen_handle) != miopenStatusSuccess) {
    fprintf(stderr, "Failed to create MIOpen handle\n");
    HIP_CLEANUP(hipStreamDestroy(state->stream));
    free(state);
    return 7;
  }

  // Set stream for MIOpen handle
  if (miopenSetStream(state->miopen_handle, state->stream) !=
      miopenStatusSuccess) {
    fprintf(stderr, "Failed to set MIOpen stream\n");
    miopenDestroy(state->miopen_handle);
    HIP_CLEANUP(hipStreamDestroy(state->stream));
    free(state);
    return 8;
  }

  // DIAGNOSTIC: Check if gcnArchName is still valid AFTER MIOpen initialization
  hipDeviceProp_t prop_after_miopen;
  if (hipGetDeviceProperties(&prop_after_miopen, 0) == hipSuccess) {
    RUNTIME_DEBUG_LOG(
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
    HIP_CLEANUP(hipStreamDestroy(state->stream));
    free(state);
    return 9;
  }

  *out_state = state;

  // Parse FlatBuffers blob to get constants_filename and constant_sizes
  if (!metadata_blob || blob_size == 0)
    return 0; // No metadata — no constants to load

  auto *meta = flatbuffers::GetRoot<mlir::hip::HipModelMetaInfo>(metadata_blob);
  auto *constants = meta->constants();
  int64_t count = constants ? (int64_t)constants->size() : 0;

  if (count <= 0)
    return 0; // No constants to load

  const char *constants_filename = "constants.bin";
  if (meta->constants_filename())
    constants_filename = meta->constants_filename()->c_str();

  // Allocate GPU constants pointer array
  state->gpu_constants =
      (void **)calloc(static_cast<size_t>(count), sizeof(void *));
  if (!state->gpu_constants) {
    fprintf(stderr, "Failed to allocate gpu_constants array\n");
    hipdnn_ep_state_cleanup(state);
    *out_state = nullptr;
    return 1;
  }
  state->num_constants = static_cast<size_t>(count);

  // Open constants file via FileSystem
  auto *fileSystem = static_cast<morphizen::FileSystem *>(fs);
  auto reader = fileSystem->create_reader_template(constants_filename);
  if (!reader) {
    fprintf(stderr, "Failed to open %s via FileSystem\n", constants_filename);
    hipdnn_ep_state_cleanup(state);
    *out_state = nullptr;
    return 1;
  }

  // Load the entire constants file as one blob, then point each
  // gpu_constants[i] into it using the offset stored in ConstantInfo.
  size_t total_size = reader->size();

  bool is_igpu = (prop.integrated == 1);

  if (is_igpu) {
    // iGPU: allocate pinned host memory. GPU reads the same physical DRAM
    // directly — no hipMemcpy needed. Draws from system RAM, not GPU quota.
    if (hipHostMalloc(&state->gpu_constants_blob, total_size,
                      hipHostMallocDefault) != hipSuccess) {
      fprintf(stderr, "hipHostMalloc failed for constants blob (%zu bytes)\n",
              total_size);
      hipdnn_ep_state_cleanup(state);
      *out_state = nullptr;
      return 1;
    }
    state->constants_blob_is_host = true;

    const void *src = reader->mmap();
    if (src) {
      memcpy(state->gpu_constants_blob, src, total_size);
    } else {
      size_t bytes_read = reader->fread(state->gpu_constants_blob, total_size);
      if (bytes_read != total_size) {
        fprintf(stderr, "Short read: got %zu of %zu bytes\n", bytes_read,
                total_size);
        hipdnn_ep_state_cleanup(state);
        *out_state = nullptr;
        return 1;
      }
    }
  } else {
    // dGPU: allocate in VRAM, upload once via hipMemcpy.
    const void *src = reader->mmap();
    void *cpu_buf = nullptr;

    if (!src) {
      // No mmap — read entire file into a staging buffer
      cpu_buf = malloc(total_size);
      if (!cpu_buf) {
        fprintf(stderr, "Failed to allocate staging buffer (%zu bytes)\n",
                total_size);
        hipdnn_ep_state_cleanup(state);
        *out_state = nullptr;
        return 1;
      }
      size_t bytes_read = reader->fread(cpu_buf, total_size);
      if (bytes_read != total_size) {
        fprintf(stderr, "Short read: got %zu of %zu bytes\n", bytes_read,
                total_size);
        free(cpu_buf);
        hipdnn_ep_state_cleanup(state);
        *out_state = nullptr;
        return 1;
      }
      src = cpu_buf;
    }

    if (hipMalloc(&state->gpu_constants_blob, total_size) != hipSuccess) {
      fprintf(stderr, "hipMalloc failed for constants blob (%zu bytes)\n",
              total_size);
      free(cpu_buf);
      hipdnn_ep_state_cleanup(state);
      *out_state = nullptr;
      return 1;
    }

    if (hipMemcpy(state->gpu_constants_blob, src, total_size,
                  hipMemcpyHostToDevice) != hipSuccess) {
      fprintf(stderr, "hipMemcpy failed for constants blob\n");
      free(cpu_buf);
      hipdnn_ep_state_cleanup(state);
      *out_state = nullptr;
      return 1;
    }

    free(cpu_buf); // no-op when mmap was used
    state->constants_blob_is_host = false;
  }

  // Point each gpu_constants[i] into the blob using the stored offset
  for (int64_t i = 0; i < count; ++i) {
    size_t offset = static_cast<size_t>(constants->Get(i)->offset());
    state->gpu_constants[i] =
        static_cast<char *>(state->gpu_constants_blob) + offset;
  }

  return 0;
}

int hipdnn_ep_state_cleanup(RuntimeState *state) {
  if (!state) {
    fprintf(stderr, "Invalid runtime state in cleanup\n");
    return 0; // Best-effort - don't fail
  }

  // Best-effort cleanup - continue even if operations fail
  // Cleanup in reverse order of initialization (LIFO)

  // Synchronize stream to ensure all GPU operations complete
  if (state->stream) {
    HIP_CLEANUP(hipStreamSynchronize(state->stream));
  }

  // Free shared workspace (if allocated)
  if (state->workspace) {
    HIP_CLEANUP(hipFree(state->workspace));
  }

  // Free memory pool (if allocated)
  if (state->pool_base) {
    HIP_CLEANUP(hipFree(state->pool_base));
  }
  if (state->buffer_offsets) {
    free(state->buffer_offsets);
  }

  // Free the single constants blob and the pointer array
  if (state->gpu_constants_blob) {
    if (state->constants_blob_is_host)
      HIP_CLEANUP(hipHostFree(state->gpu_constants_blob));
    else
      HIP_CLEANUP(hipFree(state->gpu_constants_blob));
  }
  if (state->gpu_constants)
    free(state->gpu_constants);

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
    HIP_CLEANUP(hipStreamDestroy(state->stream));
  }

  // Free the context struct itself
  free(state);

  return 0; // Best-effort cleanup always returns success
}

void *hipdnn_ep_constant_get(RuntimeState *state, int64_t index) {
  if (!state || index < 0 || (size_t)index >= state->num_constants) {
    fprintf(stderr, "hipdnn_ep_constant_get: invalid state or index %lld\n",
            (long long)index);
    return nullptr;
  }
  return state->gpu_constants[index];
}

void *hipdnn_ep_state_get_stream(RuntimeState *state) {
  return state ? static_cast<void *>(state->stream) : nullptr;
}

void *hipdnn_ep_state_get_miopen_handle(RuntimeState *state) {
  return state ? static_cast<void *>(state->miopen_handle) : nullptr;
}

void *hipdnn_ep_state_get_hipblas_handle(RuntimeState *state) {
  return state ? static_cast<void *>(state->hipblas_handle) : nullptr;
}

//==============================================================================
// Memory Pooling Support
//==============================================================================

extern "C" {

int hipdnn_ep_pool_init(RuntimeState *state, size_t pool_size,
                        const size_t *buffer_offsets, size_t num_buffers) {
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
    state->buffer_offsets = (size_t *)malloc(sizeof(size_t) * num_buffers);
    if (!state->buffer_offsets) {
      fprintf(stderr, "Failed to allocate buffer offsets array\n");
      if (state->pool_base) {
        HIP_CLEANUP(hipFree(state->pool_base));
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

void *hipdnn_ep_get_buffer_from_pool(RuntimeState *state, size_t index) {
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
  char *pool_ptr = static_cast<char *>(state->pool_base);
  size_t offset = state->buffer_offsets[index];
  return pool_ptr + offset;
}

void *hipdnn_ep_get_pool_base(RuntimeState *state) {
  if (!state) {
    fprintf(stderr, "Invalid state parameter to hipdnn_ep_get_pool_base\n");
    return nullptr;
  }
  return state->pool_base;
}

//==============================================================================
// Shared Workspace Support
//==============================================================================

void *hipdnn_ep_state_get_workspace(RuntimeState *state) {
  return state ? state->workspace : nullptr;
}

size_t hipdnn_ep_state_get_workspace_size(RuntimeState *state) {
  return state ? state->workspace_size : 0;
}

int hipdnn_ep_state_ensure_workspace(RuntimeState *state, size_t needed_size) {
  if (!state)
    return -1;
  if (needed_size == 0)
    return 0;
  if (state->workspace_size >= needed_size)
    return 0;

  // Grow: free old, allocate new
  if (state->workspace) {
    HIP_CLEANUP(hipFree(state->workspace));
    state->workspace = nullptr;
    state->workspace_size = 0;
  }

  if (hipMalloc(&state->workspace, needed_size) != hipSuccess) {
    fprintf(
        stderr,
        "hipdnn_ep_state_ensure_workspace: hipMalloc failed for %zu bytes\n",
        needed_size);
    return -1;
  }

  state->workspace_size = needed_size;
  RUNTIME_DEBUG_LOG("[workspace] Allocated shared workspace: %zu bytes\n",
                    needed_size);
  return 0;
}

} // extern "C"
