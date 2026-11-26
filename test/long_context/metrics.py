#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Numerical characterization of identical-history next-token distributions."""

import numpy as np


def compare_logits(reference, actual, token, top_k=5):
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(actual, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < 2:
        raise ValueError("Logits must have the same nontrivial vocabulary")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Non-finite logits")
    if not 0 <= token < a.size:
        raise ValueError("Reference token outside vocabulary")

    def log_softmax(x):
        y = x - x.max()
        return y - np.log(np.exp(y).sum())

    la, lb = log_softmax(a), log_softmax(b)
    pa, pb = np.exp(la), np.exp(lb)
    lm = np.logaddexp(la, lb) - np.log(2.0)
    k = min(top_k, a.size)
    ta, tb = np.argsort(a)[-k:], np.argsort(b)[-k:]
    best = np.partition(a, -2)[-2:]
    return {
        "cosine": float(
            np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)
        ),
        "relative_l2": float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-300)),
        "max_abs": float(np.max(np.abs(b - a))),
        "kl_reference_actual": float(np.sum(pa * (la - lb))),
        "js_divergence": float(0.5 * (np.sum(pa * (la - lm)) + np.sum(pb * (lb - lm)))),
        "top1_match": bool(a.argmax() == b.argmax()),
        "topk_overlap": len(set(ta) & set(tb)) / k,
        "reference_top_margin": float(best.max() - best.min()),
        "reference_token": int(token),
        "reference_nll": float(-la[token]),
        "nll_delta": float(la[token] - lb[token]),
    }


def summarize(rows):
    if not rows:
        raise ValueError("No comparisons")
    return {
        "positions": len(rows),
        "min_cosine": min(r["cosine"] for r in rows),
        "max_relative_l2": max(r["relative_l2"] for r in rows),
        "max_abs": max(r["max_abs"] for r in rows),
        "mean_kl": float(np.mean([r["kl_reference_actual"] for r in rows])),
        "mean_js": float(np.mean([r["js_divergence"] for r in rows])),
        "top1_agreement": float(np.mean([r["top1_match"] for r in rows])),
        "mean_topk_overlap": float(np.mean([r["topk_overlap"] for r in rows])),
        "mean_nll_delta": float(np.mean([r["nll_delta"] for r in rows])),
        "worst_position": max(range(len(rows)), key=lambda i: rows[i]["relative_l2"]),
    }
