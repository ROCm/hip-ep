/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_EP_DEVICE_API_H
#define HIP_EP_DEVICE_API_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(HIPDNN_EP_DEVICE_API_EXPORTS)
#define HIPDNN_EP_DEVICE_API __declspec(dllexport)
#else
#define HIPDNN_EP_DEVICE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define HIPDNN_EP_DEVICE_API __attribute__((visibility("default")))
#else
#define HIPDNN_EP_DEVICE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define HIPDNN_EP_DEVICE_API_VERSION 1u

typedef struct hipdnn_ep_state_slot_desc {
  void *base;
  uint64_t slot_bytes;
} hipdnn_ep_state_slot_desc;

enum hipdnn_ep_device_api_status {
  HIPDNN_EP_DEVICE_API_SUCCESS = 0,
  HIPDNN_EP_DEVICE_API_INVALID_ARGUMENT = 1,
  HIPDNN_EP_DEVICE_API_UNAVAILABLE = 2,
  HIPDNN_EP_DEVICE_API_LAUNCH_ERROR = 3,
};

// Returns HIPDNN_EP_DEVICE_API_VERSION. Callers must reject other versions
// before invoking any other entry point.
HIPDNN_EP_DEVICE_API uint32_t hipdnn_ep_device_api_version(void);

// Enqueues one batched state-window slot promotion on the current HIP EP
// session stream. descriptors_device points to `count` descriptors in device
// memory; descriptors_host points to the identical host-side descriptors and
// is used only for validation. Returns UNAVAILABLE when no session stream or
// matching per-architecture custom-kernel library is available.
HIPDNN_EP_DEVICE_API int hipdnn_ep_copy_state_slots(
    const hipdnn_ep_state_slot_desc *descriptors_device,
    const hipdnn_ep_state_slot_desc *descriptors_host, int32_t count,
    int32_t source_slot, int32_t destination_slot);

#ifdef __cplusplus
}
#endif

#endif // HIP_EP_DEVICE_API_H
