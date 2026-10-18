#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Independent NumPy reference for tq4_mse_qr_v1, not an inference fallback."""

import numpy as np

CODEC = "tq4_mse_qr_v1"
HEAD_DIM = 256
ROW_BYTES = HEAD_DIM // 2 + 4


def validate_constants(rotation, centroids):
    rotation = np.asarray(rotation, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    if rotation.shape != (HEAD_DIM, HEAD_DIM) or centroids.shape != (16,):
        raise ValueError("v1 requires a 256x256 rotation and 16 centroids")
    if not np.isfinite(rotation).all() or not np.isfinite(centroids).all():
        raise ValueError("nonfinite codec constants")
    if not np.all(np.diff(centroids) > 0):
        raise ValueError("centroids must be strictly increasing")
    if not np.allclose(rotation @ rotation.T, np.eye(HEAD_DIM), atol=2e-5, rtol=0):
        raise ValueError("rotation must be orthogonal")
    return rotation, centroids


def pack_indices(indices, norms):
    indices = np.asarray(indices)
    norms = np.asarray(norms, dtype="<f4")
    if indices.shape[-1] != HEAD_DIM or indices.shape[:-1] != norms.shape:
        raise ValueError("incompatible indices and norms")
    if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices > 15):
        raise ValueError("indices must be integers in [0,15]")
    if not np.isfinite(norms).all() or np.any(norms < 0):
        raise ValueError("norms must be finite and nonnegative")
    packed = indices[..., 0::2].astype(np.uint8) | (
        indices[..., 1::2].astype(np.uint8) << 4
    )
    norm_bytes = np.ascontiguousarray(norms[..., None]).view(np.uint8)
    return np.concatenate((norm_bytes, packed), axis=-1)


def unpack_rows(rows):
    rows = np.asarray(rows)
    if rows.dtype != np.uint8 or rows.shape[-1] != ROW_BYTES:
        raise ValueError("expected uint8 rows with 132 bytes")
    norms = np.ascontiguousarray(rows[..., :4]).view("<f4")[..., 0]
    indices = np.empty(rows.shape[:-1] + (HEAD_DIM,), dtype=np.uint8)
    indices[..., 0::2] = rows[..., 4:] & 15
    indices[..., 1::2] = rows[..., 4:] >> 4
    if not np.isfinite(norms).all() or np.any(norms < 0):
        raise ValueError("invalid stored norms")
    return indices, norms


def encode(x, rotation, centroids):
    rotation, centroids = validate_constants(rotation, centroids)
    x = np.asarray(x)
    if x.dtype != np.float16 or x.shape[-1] != HEAD_DIM:
        raise ValueError("v1 encodes FP16 vectors of width 256")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite input")
    x = x.astype(np.float32)
    norms = np.sqrt(np.sum(x * x, axis=-1, dtype=np.float32))
    unit = np.divide(
        x, norms[..., None], out=np.zeros_like(x), where=norms[..., None] > 0
    )
    rotated = unit @ rotation.T
    # argmin gives the lower index at an exact distance tie, matching the source.
    indices = np.abs(rotated[..., None] - centroids).argmin(axis=-1).astype(np.uint8)
    indices = np.where(norms[..., None] == 0, np.uint8(0), indices)
    return pack_indices(indices, norms)


def reconstruct(rows, rotation, centroids):
    rotation, centroids = validate_constants(rotation, centroids)
    indices, norms = unpack_rows(rows)
    return (centroids[indices] @ rotation) * norms[..., None]


def attention(q, keys, values, rotation, cb_k, cb_v, scale):
    """Dense FP64 attention over reconstructed KV, returning unrounded output."""
    q = np.asarray(q)
    if (
        q.shape != (24, HEAD_DIM)
        or keys.shape[:1] != (4,)
        or keys.shape != values.shape
    ):
        raise ValueError("expected Hq24/G4/D256 and matching K/V")
    if keys.shape[1] == 0 or not np.isfinite(scale):
        raise ValueError("nonempty cache and finite attention scale required")
    k = reconstruct(keys, rotation, cb_k).astype(np.float64)
    v = reconstruct(values, rotation, cb_v).astype(np.float64)
    result = np.empty((24, HEAD_DIM), dtype=np.float64)
    for h in range(24):
        scores = (k[h // 6] @ q[h].astype(np.float64)) * scale
        weights = np.exp(scores - scores.max())
        result[h] = (weights @ v[h // 6]) / weights.sum()
    return result


def partitioned_attention(q, keys, values, rotation, cb_k, cb_v, scale, partitions):
    """Rotated-frame FP32 partition reference with unnormalized numerators."""
    if partitions < 1 or keys.shape[1] == 0:
        raise ValueError("positive partitions and nonempty cache required")
    ki, kn = unpack_rows(keys)
    vi, vn = unpack_rows(values)
    qr = np.asarray(q, np.float32) @ rotation.T
    output = np.empty((24, HEAD_DIM), dtype=np.float32)
    for h in range(24):
        g = h // 6
        states = []
        for positions in np.array_split(np.arange(keys.shape[1]), partitions):
            if positions.size == 0:
                continue
            scores = (cb_k[ki[g, positions]] @ qr[h]) * kn[g, positions] * scale
            maximum = scores.max()
            weights = np.exp(scores - maximum)
            numerator = (weights * vn[g, positions]) @ cb_v[vi[g, positions]]
            states.append((maximum, weights.sum(dtype=np.float32), numerator))
        maximum = max(s[0] for s in states)
        denominator = np.float32(0)
        numerator = np.zeros(HEAD_DIM, np.float32)
        for m, length, acc in states:
            factor = np.exp(m - maximum)
            denominator += factor * length
            numerator += factor * acc
        output[h] = (numerator / denominator) @ rotation
    return output
