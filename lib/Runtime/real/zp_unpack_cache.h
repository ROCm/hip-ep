/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_REAL_ZP_UNPACK_CACHE_H
#define HIPDNN_EP_REAL_ZP_UNPACK_CACHE_H

// Per-RuntimeState pointer-keyed cache for unpacked MatMulNBits zero_points
// buffers. Used by both wrap_matmul_nbits and wrap_qmoe to avoid re-launching
// the unpack/convert kernels on every call when zero_points pointers are
// stable across inferences (which they are — the pointers come from the
// model constants blob).
//
// The cache itself (ZpUnpackCache) lives in matmul_nbits.cpp; this header
// just exposes the two lookup helpers and the global destroyer.

struct RuntimeState;

namespace hipdnn_ep_real {

// Returns the cached uint8 zero_points buffer for `zp_packed`, allocating +
// unpacking on miss. Returns nullptr only on hipMalloc failure.
const void *lookup_or_unpack_zp_u8(RuntimeState *state, void *stream,
                                   const void *zp_packed, int N, int groups_k);

// Returns the cached fp16 zero_points buffer for `zp_packed`, allocating +
// converting on miss. Returns nullptr only on hipMalloc failure.
const void *lookup_or_convert_zp_fp16(RuntimeState *state, void *stream,
                                      const void *zp_packed, int N,
                                      int groups_k);

} // namespace hipdnn_ep_real

#endif // HIPDNN_EP_REAL_ZP_UNPACK_CACHE_H
