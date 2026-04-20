#include "mm_hal.h"

#ifdef MM_HAS_HIP

#include <cstring>
#include <hip/hip_runtime.h>

/* ROCm/HIP HAL backend — wraps HIP runtime API. */

static int hip_get_device_count(void) {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess)
    return 0;
  return count;
}

static int hip_get_device_info(int device_id, mm_device_info_t *info) {
  if (!info)
    return MM_ERROR_INVALID_ARG;

  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, device_id) != hipSuccess)
    return MM_ERROR_HAL_FAILURE;

  std::strncpy(info->name, props.name, MM_DEVICE_NAME_MAX - 1);
  info->name[MM_DEVICE_NAME_MAX - 1] = '\0';
  info->type = (mm_device_t)device_id;
  info->total_memory = props.totalGlobalMem;

  size_t free_mem = 0, total_mem = 0;
  int prev_device = 0;
  hipGetDevice(&prev_device);
  hipSetDevice(device_id);
  hipMemGetInfo(&free_mem, &total_mem);
  hipSetDevice(prev_device);
  info->free_memory = free_mem;

  return MM_OK;
}

static void *hip_raw_alloc(int device_id, size_t size, size_t /*alignment*/) {
  if (size == 0)
    return nullptr;

  int prev_device = 0;
  hipGetDevice(&prev_device);
  hipSetDevice(device_id);

  void *ptr = nullptr;
  hipError_t err = hipMalloc(&ptr, size);

  hipSetDevice(prev_device);
  return (err == hipSuccess) ? ptr : nullptr;
}

static void hip_raw_free(int device_id, void *ptr) {
  if (!ptr)
    return;
  int prev_device = 0;
  hipGetDevice(&prev_device);
  hipSetDevice(device_id);
  hipFree(ptr);
  hipSetDevice(prev_device);
}

static void *hip_stream_alloc(int device_id, size_t size, mm_stream_t stream) {
  if (size == 0)
    return nullptr;

  int prev_device = 0;
  hipGetDevice(&prev_device);
  hipSetDevice(device_id);

  void *ptr = nullptr;
  hipError_t err =
      hipMallocAsync(&ptr, size, reinterpret_cast<hipStream_t>(stream));
  if (err != hipSuccess) {
    /* Fall back to synchronous alloc */
    err = hipMalloc(&ptr, size);
  }

  hipSetDevice(prev_device);
  return (err == hipSuccess) ? ptr : nullptr;
}

static void hip_stream_free(int device_id, void *ptr, mm_stream_t stream) {
  if (!ptr)
    return;
  int prev_device = 0;
  hipGetDevice(&prev_device);
  hipSetDevice(device_id);

  hipError_t err = hipFreeAsync(ptr, reinterpret_cast<hipStream_t>(stream));
  if (err != hipSuccess) {
    hipFree(ptr);
  }

  hipSetDevice(prev_device);
}

static int hip_async_copy(void *dst, const void *src, size_t size,
                          mm_copy_kind_t kind, mm_stream_t stream,
                          mm_fence_t * /*fence*/) {
  hipMemcpyKind hk;
  switch (kind) {
  case MM_COPY_HOST_TO_DEVICE:
    hk = hipMemcpyHostToDevice;
    break;
  case MM_COPY_DEVICE_TO_HOST:
    hk = hipMemcpyDeviceToHost;
    break;
  case MM_COPY_DEVICE_TO_DEVICE:
    hk = hipMemcpyDeviceToDevice;
    break;
  case MM_COPY_HOST_TO_HOST:
    hk = hipMemcpyHostToHost;
    break;
  default:
    return MM_ERROR_INVALID_ARG;
  }

  hipError_t err =
      hipMemcpyAsync(dst, src, size, hk, reinterpret_cast<hipStream_t>(stream));
  return (err == hipSuccess) ? MM_OK : MM_ERROR_HAL_FAILURE;
}

static int hip_memset(void *ptr, int value, size_t size, mm_stream_t stream) {
  hipError_t err =
      hipMemsetAsync(ptr, value, size, reinterpret_cast<hipStream_t>(stream));
  return (err == hipSuccess) ? MM_OK : MM_ERROR_HAL_FAILURE;
}

static int hip_stream_sync(mm_stream_t stream) {
  hipError_t err = hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream));
  return (err == hipSuccess) ? MM_OK : MM_ERROR_HAL_FAILURE;
}

static size_t hip_get_total_memory(int device_id) {
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, device_id) != hipSuccess)
    return 0;
  return props.totalGlobalMem;
}

static size_t hip_get_free_memory(int device_id) {
  size_t free_mem = 0, total_mem = 0;
  int prev_device = 0;
  hipGetDevice(&prev_device);
  hipSetDevice(device_id);
  hipMemGetInfo(&free_mem, &total_mem);
  hipSetDevice(prev_device);
  return free_mem;
}

static void *hip_host_alloc(size_t size, size_t /*alignment*/) {
  if (size == 0)
    return nullptr;
  void *ptr = nullptr;
  hipError_t err = hipHostMalloc(&ptr, size);
  return (err == hipSuccess) ? ptr : nullptr;
}

static void hip_host_free(void *ptr) {
  if (ptr)
    hipHostFree(ptr);
}

static const mm_hal_t g_hip_hal = {
    hip_get_device_count, hip_get_device_info, hip_raw_alloc,
    hip_raw_free,         hip_stream_alloc,    hip_stream_free,
    hip_async_copy,       hip_memset,          hip_stream_sync,
    hip_get_total_memory, hip_get_free_memory, hip_host_alloc,
    hip_host_free,
};

const mm_hal_t *mm_hal_hip_get(void) { return &g_hip_hal; }

#endif /* MM_HAS_HIP */
