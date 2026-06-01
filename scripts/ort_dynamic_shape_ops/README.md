<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Output allocation for data-dependent-shape ops (`Range`, `ConstantOfShape`, `NonZero`)

This note explains **who allocates the output tensor** for three ONNX ops
whose output shape is *data-dependent* — it depends on the input **values /
data**, not just the input shapes — and traces the behaviour to its **root
cause in the ONNX Runtime source**.

- **Verification scripts:** `verify_range.py`, `verify_constantofshape.py`,
  `verify_nonzero.py`, driven by `common.py`; run everything with
  `python run_all.py`.
- **Execution provider:** `providers=["CPUExecutionProvider"]` (CPU EP).
- **Source traced against:** ONNX Runtime
  [`main`](https://github.com/microsoft/onnxruntime/tree/main), pinned for
  line-accurate links to commit
  [`13af659`](https://github.com/microsoft/onnxruntime/commit/13af65970aaa6a0b9ac71106da07376fef24aa56)
  (2026-05-31; latest release: 1.26.0). Every code snippet below names its
  **exact file path, function, and line range** at this commit, and the key
  lines carry inline `// :<line>` markers so they are trivial to locate. All
  links are also collected in [§ Source references](#source-references).

---

## TL;DR

**You do not need to pre-allocate the output for these ops. ONNX Runtime
allocates it, sized from the shape the kernel computes at run time.**

| How you call it | Who allocates | Result |
| --- | --- | --- |
| `session.run(None, feeds)` | **ORT** | Correctly-sized output returned as numpy; you pass nothing. |
| `IOBinding` + `bind_output(name, "cpu")` (no buffer) | **ORT** | Correctly-sized output; you state only the *device*, never the size. |
| `IOBinding` + `bind_ortvalue_output(name, buf)`, **exact** size | caller | ORT **reuses** your buffer (same data pointer). |
| `IOBinding` + pre-bound buffer, **wrong** size | — | ORT **raises** `INVALID_ARGUMENT`; it never reallocates or truncates. |

Because the output shape changes per run, **a single fixed pre-allocated
buffer cannot serve these ops once the shape moves** — it fits only the run
whose shape it happens to match and is rejected for the others. Practical
rule: let ORT allocate, or re-request the exact shape every run.

---

## Why these outputs are data-dependent

Each kernel reads its inputs, computes the output `TensorShape` **at compute
time**, and asks the framework for an output of exactly that shape via
`ctx->Output(0, shape)`. Static shape inference therefore leaves the relevant
dims symbolic (`?`), e.g. `Range -> [range_len]`, `ConstantOfShape ->
[d0, d1]` (or unknown rank), `NonZero -> [2, num_nonzero]`.

**`Range`** — length is a function of the scalar *values*.
File `onnxruntime/core/providers/cpu/generator/range.cc`, function
`ComputeRange<T>`, lines
[57–79](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/providers/cpu/generator/range.cc#L57-L79):

```cpp
int64_t n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));  // :68 length from VALUES
if (n <= 0) n = 0;                                                        // :69-70
TensorShape shape = {n};                                                  // :71
T* y = ctx->Output(0, shape)->MutableData<T>();                           // :72 request value-sized output
```

**`ConstantOfShape`** — output shape *is* the input tensor's values; even the
rank is data-dependent.
File `onnxruntime/core/providers/cpu/generator/constant_of_shape_base.h`,
function `PrepareCompute`, lines
[106–122](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/providers/cpu/generator/constant_of_shape_base.h#L106-L122):

```cpp
const auto shape_tensor = ctx->template Input<Tensor>(0);          // :108 the shape INPUT tensor
const auto span = shape_tensor->template DataAsSpan<int64_t>();    // :116 its VALUES
TensorShape output_shape(span);                                    // :118 become the output dims
(*output_tensor) = ctx->Output(0, output_shape);                   // :119 request value-sized output
```

**`NonZero`** — column count `N` is the number of non-zero elements, known
only after scanning the data.
File `onnxruntime/core/providers/cpu/tensor/nonzero_op.cc`, function
`NonZero<T>::Compute`, lines
[101–108](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/providers/cpu/tensor/nonzero_op.cc#L101-L108):

```cpp
const Eigen::Index num_non_zero_values =                                       // :101 N from DATA
    narrow<Eigen::Index>(non_zero_indices_buffer.size()) / coordinate_size;
Tensor* const Y = context->Output(0, {coordinate_size, num_non_zero_values});  // :108 request [rank, N]
```

---

## Root cause in the ONNX Runtime source

The output of `ctx->Output(0, shape)` flows through one decision point. That
point is the entire reason for every row in the TL;DR table.

### 1. The kernel's `Output(...)` forwards the computed shape

`OpKernelContext::Output` → `OutputMLValue` → the execution frame.
File `onnxruntime/core/framework/op_kernel.cc`, lines
[44–85](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/framework/op_kernel.cc#L44-L85):

```cpp
Tensor* OpKernelContext::Output(int index, const TensorShape& shape) {  // :44
  auto p_ml_value = OutputMLValue(index, shape);
  return p_ml_value ? p_ml_value->GetMutable<Tensor>() : nullptr;
}

OrtValue* OpKernelContext::OutputMLValue(int index, const TensorShape& shape) {  // :72
  ...
  Status status = execution_frame_->GetOrCreateNodeOutputMLValue(               // :82 forward computed shape
      index, GetOutputArgIndex(index), &shape, p_ml_value, kernel_->Node());
  ...
}
```

### 2. `GetOrCreateNodeOutputMLValue` decides allocate-vs-reuse-vs-reject

File `onnxruntime/core/framework/execution_frame.cc`, function
`IExecutionFrame::GetOrCreateNodeOutputMLValue`, lines
[144–220](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/framework/execution_frame.cc#L144-L220).
The slot for this output is `all_values_[ort_value_idx]`; whether it is
already allocated decides everything (line markers below are the real source
lines at the pinned commit):

```cpp
p_ort_value = &all_values_[ort_value_idx];               // :154

if (p_ort_value->IsAllocated()) {                        // :156 caller pre-bound a buffer
  // :157-176  comment block: reuse when shapes match, reject when they differ
  bool shape_matched = true;                             // :177
  if (p_ort_value->IsTensor()) {                         // :179
    const Tensor& tensor = p_ort_value->Get<Tensor>();   // :181
    shape_matched = (tensor.Shape() == *shape);          // :182 (A) compare
  }
  if (!shape_matched) {                                  // :191
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,            // :199 (B) reject
        "The output OrtValue provided for output '", ...,
        "' has shape ", existing_shape,
        " but the computed output shape for this run is ", *shape, ...);
  }
} else {                                                 // :210
  if (shape != nullptr && IsOutput(ort_value_idx))       // :212
    VerifyOutputSizes(output_index, node, *shape);       // :213 (C) warn-only check
  status = CreateNodeOutputMLValueImpl(*p_ort_value, ort_value_idx, shape);  // :215 (D) ALLOCATE
}
```

This single function explains all observed behaviour:

- **`session.run` / `bind_output(device)` → branch (D).** The frame's slot is
  *not* allocated (no caller buffer), so `CreateNodeOutputMLValueImpl`
  allocates a fresh buffer using the kernel-computed `shape` (and, for an
  IOBinding device binding, the bound allocator). **This is "ORT allocates for
  you."**
- **Exact-size pre-bound buffer → (A) matches → reuse.** The slot was filled
  at frame `Init()` from the caller's fetches/IOBinding outputs;
  `tensor.Shape() == *shape` is true, so ORT keeps the caller's buffer. This
  is why the scripts see the **data pointer preserved** (`caller_buf ptr ==
  result ptr`).
- **Wrong-size pre-bound buffer → (A) fails → (B) reject.** ORT returns
  `INVALID_ARGUMENT`; it does **not** reallocate or truncate. This is the
  error the scripts catch for the too-small / too-large / shape-changed cases.

### 3. `VerifyOutputSizes` is only a warning (not the rejection)

Note the rejection in (B) is distinct from `VerifyOutputSizes` (branch C),
which compares the kernel's shape against the model's *declared* shape and
only **logs a warning** when a fixed declared dim disagrees.
File `onnxruntime/core/framework/execution_frame.cc`, function
`ExecutionFrame::VerifyOutputSizes`, lines
[892–917](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/framework/execution_frame.cc#L892-L917):

```cpp
void ExecutionFrame::VerifyOutputSizes(int output_index, const Node& node,    // :892
                                       const TensorShape& output_shape) {
  const auto* expected_shape = output_def->Shape();                           // :894
  if (expected_shape == nullptr) return;   // :895 unknown/dynamic -> nothing to check
  ...
  if (!compatible) {                                                          // :912
    LOGS(..., WARNING) << "Expected shape from model of " << ...              // :913
                       << " does not match actual shape of " << output_shape ...;
  }
}
```

So a symbolic/dynamic declared dim sails through (nothing to compare), and a
freshly-allocated output is always sized to the kernel's runtime shape.

---

## Source references

All links are pinned to ONNX Runtime `main` @
[`13af659`](https://github.com/microsoft/onnxruntime/commit/13af65970aaa6a0b9ac71106da07376fef24aa56)
(2026-05-31) so the line anchors stay valid. Browse the live tree at
[`microsoft/onnxruntime` `/tree/main`](https://github.com/microsoft/onnxruntime/tree/main).

| What it shows | File · lines |
| --- | --- |
| `OpKernelContext::Output` → `OutputMLValue` forwards the kernel's computed shape to the frame | [`op_kernel.cc` L44–L85](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/framework/op_kernel.cc#L44-L85) |
| `GetOrCreateNodeOutputMLValue` — the allocate (D) / reuse (A) / reject (B) decision | [`execution_frame.cc` L144–L220](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/framework/execution_frame.cc#L144-L220) |
| `VerifyOutputSizes` — declared-vs-actual shape check (warning only) | [`execution_frame.cc` L892–L917](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/framework/execution_frame.cc#L892-L917) |
| `Range` kernel computes `n` from scalar values, then `ctx->Output(0, {n})` | [`range.cc` L57–L79](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/providers/cpu/generator/range.cc#L57-L79) |
| `ConstantOfShape` builds `output_shape` from input *values*, then `ctx->Output` | [`constant_of_shape_base.h` L106–L122](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/providers/cpu/generator/constant_of_shape_base.h#L106-L122) |
| `NonZero` counts non-zeros, then `context->Output(0, {rank, N})` | [`nonzero_op.cc` L101–L108](https://github.com/microsoft/onnxruntime/blob/13af65970aaa6a0b9ac71106da07376fef24aa56/onnxruntime/core/providers/cpu/tensor/nonzero_op.cc#L101-L108) |

> The snippets in the sections above are lightly edited for readability
> (shortened SSA/variable names, elided error-handling and attribute clutter).
> Follow a link for the verbatim source at the pinned commit.

---

## Reproduce

```bash
python run_all.py                 # all three ops, full battery + dynamic-shape block
python verify_range.py            # one op at a time
python verify_constantofshape.py
python verify_nonzero.py
```

Each op prints: static shape inference (symbolic dims), the three calling
conventions, the pre-allocated-buffer probe, and a "dynamic-shape input"
block that drives one session with several runtime shapes (including the
`NonZero` `N=0` empty-output edge case and `ConstantOfShape`'s rank changing
per run).
