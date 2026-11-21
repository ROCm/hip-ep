#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""CPU oracle for frozen Qwen selector features and stable causal block support."""

import torch
import torch.nn.functional as functional


def norm(value, weight):
    floating = value.float()
    scaled = floating * torch.rsqrt(floating.square().mean(-1, keepdim=True) + 1e-6)
    return (scaled * (1 + weight.float())).to(value.dtype)


def rotate(value, cos, sin):
    assert cos.shape[-1] == sin.shape[-1] == 64
    while cos.ndim < value.ndim:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    first = value[..., :64].float()
    swapped = torch.cat((-first[..., 32:], first[..., :32]), -1)
    rotated = first * cos.float() + swapped * sin.float()
    return torch.cat((rotated.to(value.dtype), value[..., 64:]), -1)


@torch.inference_mode()
def features(hidden, weights, cos, sin):
    assert hidden.device.type == "cpu" and hidden.shape[-1] == 5120
    projected = functional.linear(
        hidden.to(torch.bfloat16), weights["projection.weight"]
    )
    query, key = projected.split([512, 128], -1)
    query = norm(query.reshape(*hidden.shape[:2], 4, 128), weights["q_norm.weight"])
    query = rotate(query, cos, sin)
    key = norm(key, weights["token_k_norm.weight"])
    batch, length, width = key.shape
    groups = length // 4
    pooled = (
        key[:, : groups * 4]
        .float()
        .reshape(batch, groups, 4, width)
        .mean(2)
        .to(torch.bfloat16)
    )
    pooled = norm(pooled, weights["pool_k_norm.weight"])
    pooled = rotate(pooled, cos[:, : groups * 4 : 4], sin[:, : groups * 4 : 4])
    return {
        "projected": projected,
        "query": query,
        "token_key": key,
        "pooled_key": pooled,
    }


def scores(query, pooled):
    """Match source's 16-lane dot reduction and sum of four nonnegative heads."""
    assert query.shape == (4, 128) and pooled.ndim == 2 and pooled.shape[1] == 128
    partial = torch.zeros(pooled.shape[0], 4, 16, dtype=torch.float32)
    for first in range(0, 128, 16):
        partial += (
            pooled[:, None, first : first + 16].float()
            * query[None, :, first : first + 16].float()
        )
    for offset in [8, 4, 2, 1]:
        partial[:, :, :offset] = (
            partial[:, :, :offset] + partial[:, :, offset : 2 * offset]
        )
    result = torch.zeros(pooled.shape[0], dtype=torch.float32)
    for head in range(4):
        result += partial[:, head, 0].clamp_min(0)
    return result * 0.08838834764831845


def select(query, pooled, position):
    historical = position // 4
    assert historical <= pooled.shape[0]
    selected = torch.full((513,), -1, dtype=torch.int32)
    if historical <= 512:
        selected[: historical + 1] = torch.arange(historical + 1, dtype=torch.int32)
    else:
        values = scores(query, pooled[:historical])
        ranked = torch.argsort(values, descending=True, stable=True)[:512]
        selected[:512] = ranked.sort().values.to(torch.int32)
        selected[512] = historical
    return selected


def token_support(selected, position):
    result = [
        4 * int(group) + offset
        for group in selected
        if group >= 0
        for offset in range(4)
        if 4 * int(group) + offset <= position
    ]
    assert len(result) <= 2052 and len(set(result)) == len(result)
    assert result == sorted(result) and result[-1] == position
    return result
