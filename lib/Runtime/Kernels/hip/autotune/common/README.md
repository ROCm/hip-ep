<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Shared autotune vocabulary

One dtype enum, three consumers, so the numbering can never drift between an
op's `.fbs` schema, its C++ dispatch code, and its `update_lut.py`:

```
hipdnn_dtype.fbs   enum HipdnnDType : ubyte   -- the flatbuffers source of truth
hipdnn_dtype.h     hipdnn_ep::common::HipdnnDType + toString() -- hand-kept mirror
dtype.py           NAME_TO_VALUE / VALUE_TO_NAME dicts -- for update_lut.py scripts
```

`gemm_autotune.fbs` is the first consumer (`include "../common/hipdnn_dtype.fbs";`).
`matmul_nbits`'s equivalent migration is planned, not yet done -- see
`../gemm/plan.md` §3 and `../matmul_nbits/README.md`.

## The one rule: only append, never renumber

```
enum HipdnnDType : ubyte {
  Any = 0,      // a memset field reads as "unclassified" and is rejected at load
  F32 = 1, F16 = 2, BF16 = 3, F64 = 4,
  I32 = 5, I8 = 6, U8 = 7,
  I4 = 8, U4 = 9, U3 = 10, U2 = 11,
  F8E4M3 = 12, F8E5M2 = 13,
}
```

An already-compiled `.fb` in the field encodes these numbers. Renumbering (or
reusing a value that turned out to be a mistake) silently reinterprets every
existing table's dtype fields as a *different* dtype -- there is no schema
check that would catch this, since the field would still parse as a valid
enum value. Add a new value at the end instead, always.

Floating-point and sub-byte-quantized-weight encodings (`U4`/`U3`/`U2`/`I4`)
share one enum on purpose, even though they read very differently on the
wire: they occupy the *same field position* (typically `wts_dtype`) across
every op's schema, so splitting them into two enums would need two fields
everywhere instead of one.

## `hipdnn_dtype.h` is a hand-kept mirror, not generated

`gemm_kernel.hip` (and any other `.hip` kernel file) needs to name a dtype
without pulling in the flatc-generated header, so `hipdnn_dtype.h` duplicates
the enum as a plain `enum class`. Keep the two in lock-step by hand -- it is
a tiny, append-only list, so this is a rare, mechanical edit, not a
maintenance burden. `toString()` is what the debug logs
(`HIPDNN_EP_DEBUG=1` / `HIPDNN_GEMM_LUT_LOG=1`) print, e.g. `act=f16 wts=f16
out=f16` -- three-dtype tables are unreadable if this prints numbers instead
(that was v1's `type_bytes` problem).

`dtype.py`'s `dtype_value()` / `dtype_name()` raise on an unknown token
rather than silently mapping it to `Any` -- a typo'd dtype name in a sweep
log should fail a `build` loudly, not quietly produce an unclassified point
that can never be queried back out.
