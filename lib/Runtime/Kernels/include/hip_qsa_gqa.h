/* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License. */
#ifndef HIP_EP_QSA_GQA_H
#define HIP_EP_QSA_GQA_H
#include "hip_qsa.h"
#ifdef __cplusplus
extern "C" {
#endif
HIP_KERNEL_API int hip_qsa_cast_hidden(void *stream, const void *hidden,
                                     void *bf16, int sequence);
HIP_KERNEL_API int hip_qsa_project_decode(void *stream, const void *hidden,
                                        const void *weight, void *projected);
#ifdef __cplusplus
}
#endif
#endif
