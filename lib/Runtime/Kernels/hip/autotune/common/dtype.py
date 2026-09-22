"""Shared dtype vocabulary for hipdnn-ep offline autotune LUTs.

Python mirror of hipdnn_dtype.fbs / hipdnn_dtype.h -- one enum, three
consumers (the .fbs, the C++ loader, this file for update_lut.py). Rule: only
append; never renumber or reuse a value.
"""
from __future__ import annotations

NAME_TO_VALUE = {
    "any": 0,
    "f32": 1,
    "f16": 2,
    "bf16": 3,
    "f64": 4,
    "i32": 5,
    "i8": 6,
    "u8": 7,
    "i4": 8,
    "u4": 9,
    "u3": 10,
    "u2": 11,
    "f8e4m3": 12,
    "f8e5m2": 13,
}

VALUE_TO_NAME = {v: k for k, v in NAME_TO_VALUE.items()}


def dtype_value(name: str) -> int:
    key = name.strip().lower()
    if key not in NAME_TO_VALUE:
        raise ValueError(f"unknown HipdnnDType name {name!r}")
    return NAME_TO_VALUE[key]


def dtype_name(value: int) -> str:
    if value not in VALUE_TO_NAME:
        raise ValueError(f"unknown HipdnnDType value {value!r}")
    return VALUE_TO_NAME[value]
