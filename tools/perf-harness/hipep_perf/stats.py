#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Small, dependency-free statistics for paired A/B results."""

from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass


def _betacf(a: float, b: float, x: float) -> float:
    # Lentz continued fraction for the regularized incomplete beta function.
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _ibeta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * _ibeta(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_crit(df: int, confidence: float = 0.95) -> float:
    """Two-sided critical value of Student's t, by bisection on the CDF."""
    if df < 1:
        raise ValueError("df must be >= 1")
    target = 1.0 - (1.0 - confidence) / 2.0
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass
class Paired:
    n: int
    mean_diff: float
    sd_diff: float
    half_width: float
    pct: float

    @property
    def low(self) -> float:
        return self.mean_diff - self.half_width

    @property
    def high(self) -> float:
        return self.mean_diff + self.half_width

    @property
    def spans_zero(self) -> bool:
        return self.low * self.high <= 0


def paired(base: list[float], arm: list[float], confidence: float = 0.95) -> Paired:
    """Interval over per-round differences (arm - base). Rounds must be aligned."""
    if len(base) != len(arm) or len(base) < 2:
        raise ValueError("need >= 2 aligned rounds")
    diffs = [a - b for a, b in zip(arm, base)]
    m, sd = st.mean(diffs), st.stdev(diffs)
    half = t_crit(len(diffs) - 1, confidence) * sd / math.sqrt(len(diffs))
    return Paired(len(diffs), m, sd, half, 100.0 * m / st.mean(base))
