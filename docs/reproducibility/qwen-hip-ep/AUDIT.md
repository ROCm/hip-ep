<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Audit of the native QSA change

Scope: the complete local OLMo+Qwen stack relative to local`origin/main`, including
custom kernels, graph export, BF16 import, compiler lowering, per-node state,
shared dense GQA changes and experiment records. Upstream-only changes are not
claimed as part of QSA. The isolated PR checkout merges the local main reference;
the measured install remains unchanged. No remote update is part of this work.

## Findings and corrections

| Finding | Resolution |
|---|---|
| Seven merge conflicts with local main: GQA host-length/bias ABI, registration, BF16/FP8 import, EP compute tracing/error translation | Preserve upstream arguments/ROCTx/FP8 and QSA child cache/BF16/exception handling; adapt the child dense call to the new argument order |
| Main's DPS interface now requires explicit result-shape reification | Move both QSA operations to the shared `Hip_DpsOp_AutoReify` family, matching ordinary GQA |
| Main's Hipsr parent-edge code assumes a newer LLVM region-successor API than the cached toolchain | Add a parent-edge constructor adapter for both APIs; use the existing successor-region accessor for exit detection |
| OLMo malformed int32 sequence length could overflow in`length+1-S` before validation | Compute in int64, validate, then narrow; require an error flag before dispatch |
| Mock QSA stubs returned an unsupported-operation code that lowering could discard | Set the mock's host-backed error flag as well; validate the complete mock source with MSVC syntax checking |
| OLMo runtime and included kernel ABI headers were missing explicit custom-command dependencies | Track the runtime header and configure-time kernel ABI digest so header edits rebuild their consumers on Windows |
| Qwen design introduction still described a queued port | Update to integrated status and link measured evidence |
| Inspector recorded unused`gqa_back.cpp` | Correct future inspector to compiled`gqa.cpp`; retain historical JSON and erratum |
| Generic Qwen stage receipts hashed the OLMo payload and snapshotted only OLMo scripts | Preserve the historical receipts with an erratum; Qwen execution/export protocols remain the authoritative identity. Future stage runner selects the Qwen payload and snapshots both script directories |
| Stage interruption could leave child processes outside the timeout/RAM failure paths | Stop owned descendants on any interruption and verify ownership before removing the GPU-stage lock |
| Existing incremental primitive test covered only13 steps | Extend to137 single-token updates and exact full-prefill pool equality |
| Existing long ranking controls used only zero-score ties | Add seeded nonzero integer BF16 ranking controls at positions32771/65599 with exact independent stable CPU indices |
| Qwen error recovery lacked a complete-operator receipt | Add decode-before-prefill, extreme/negative length, bad-position and next-prefill recovery controls |
| Memory numbers could be mistaken for physical residency or sparse KV savings | Label WDDM commitment, phase sampling limits and full KV retention; export complete/pending/bounded statuses |
| Final branch commit does not identify the historically measured binary | Preserve binary SHA256, original receipts and explicit source reconstruction/build-provenance limit |
| Git's automatic line-ending conversion would invalidate raw Windows protocol hashes on another OS | Preserve packet JSON/CSV/raw bytes with scoped attributes and recognize CR as part of its original line ending |
| Text-fixing hooks would rewrite frozen raw evidence, including valid JSON without a final newline | Exclude only the packet's `raw/` tree from mutating whitespace/BOM/line-ending hooks; keep large-file and other checks active |

## Kernel and runtime review

Qwen review traced FP16-to-BF16 projection input rounding, prefill hipBLASLt and
single-token projection, three zero-centered norms, token-normalize-before-pool,
BF16 pool boundary, half-rotary coordinates and block-start positions. Ranking
sums four rectified head scores, uses nonnegative float radix order, chooses
earlier indices on ties and emits ascending historical IDs plus one causal
current block. Attention uses native Q/K/V, head mapping Q/6, scale1/16, bounded
support and FP32 softmax/split merging. Shuffles, block scans, shared-memory
barriers, tail completion, batch strides and int64 cache offsets were inspected.

OLMo remains a separate specialization with its own pooling/current-block
semantics,128-group budget, width4096 and batch1 constraint. Its math must not be
silently replaced with the Qwen contract. The shared dense GQA addition is an
owned child GEMM cache used by Qwen's full-support path; ordinary GQA keeps its
default cache path. The merge preserves upstream host length and mask extents.

Each specialization owns scratch, GEMM descriptors, pooled keys/tail and cache
history in its node state. Prefill resets history; decode validates prior length,
capacity and in-place KV pointers. Failures set the EP device error flag; the
Compute boundary converts exceptions to OrtStatus. Geometry, canonical text
positions, architecture and uniform batch guards are part of the contract.
Raw launcher APIs assume correctly sized contiguous device buffers and the
qualified wave size; they are internal APIs, not general tensor-validation APIs.

The compiler review includes custom schema registration, ONNX conversion, DPS
outputs/aliasing, verifier dtype/shape constraints, bufferization, runtime ABI,
state construction/destruction and real/mock symbol registration. BF16 import
support coexists with main's FP8 support. No change trains weights or substitutes
selector scores for attention logits. No history-gather buffer is introduced.

## Measurement and statistical review

Original/Changed exports preserve all other node and initializer bytes, share
the same reconstructed backbone and use the same cache capacities/head. Historical
controls establish full-support equality, pruned full-vs-cached numerical
agreement, stable replay and one EP dispatch per forward. The quality auditor
recomputes PPL/answer extraction from raw rows, checks document/question IDs,
tokenizer/chat-template identity, protocol hashes, caps and paired variants.
The performance auditor recomputes all phase TPS, balanced pair ordering, means
and seeded bootstrap intervals and verifies binary/stream/protocol hashes.

Timing includes GPU synchronization and the full session. Allocations/bindings,
state zeroing and token sampling are outside its phase timers. This is an
end-to-end session comparison, not a standalone attention-kernel speedup. Six
pairs do not establish cross-run repeatability. The memory sampler joined32K
after four measured pairs; this instrumentation difference is retained. No64K
timing is valid. Higher batches have memory probes only, not headline throughput.

Memory probes isolate variants in fresh processes and keep exact source snapshots
and all phase traces. They measure first-pass allocation plus a short decode;
they do not provide steady-state throughput. The batch4 first-pass delay remains
unexplained and is not labeled a kernel hang or successful performance result.
Incomplete probes are lower bounds only. Full KV and per-node scratch explain
why QSA can use more memory despite attending fewer tokens.

## Validation record

Historical measured-stack receipts cover611 passing compiler lit tests
(9 unsupported), three compiler/runtime unit tests, primitive CPU-oracle math,
batch4/8 feature controls, long tie controls, complete operator projection and
attention, full-model numerical/replay/dispatch controls and matched quality.
Historical receipts remain under`raw/` and in the local raw archive; their hashes
are not rewritten to match new scripts.

The merged candidate at source commit
`b70ce2829ad58d145ee0cb6be942d0c3a51df573` passes **736 compiler lit tests**
(nine unsupported), **eight CPU unit tests** (six default plus BF16 mapping and
constant externalization), and the expanded kernel, complete-operator and
full-model controls. [candidate-validation.json](candidate-validation.json)
binds these checks to source/binary hashes, commands, logs and raw receipts.
The export-invariance suite also passes six CPU tests.

On the candidate, full-support outputs are bit-exact between variants; sparse
full-vs-cached relative RMS is 0.082021%, first-pruning continuity is 0.106473%,
and replay is exact. Original has 20 forwards/20 EP dispatches; Changed has
39/39. Cache pointers remain stable. These establish correctness of the merge;
the archived performance/accuracy campaign belongs to the measured binaries.
Publication must retain canceled memory-cell statuses and the absence of matched
QSA natural-long quality, Qwen BFCL and Qwen SWE-bench.

The expanded measured-binary controls also pass: all four selectors preserve
exact pool state through 137 decode steps; nonzero long-history rankings match
stable CPU indices; a single KV append at position 65599 uses the correct cache
stride; malformed requests are rejected and a subsequent prefill recovers.
This last primitive append is not a 64K full-model retry or measurement.

The candidate build disables main's unrelated package-wide kernel test executable
targets after their object paths exceeded CMake's Windows path-length guidance.
Compiler/unit suites and the dedicated QSA GPU checks remain the validation path.
The long local Windows path also requires short, matching MorphiZen internal
target IDs; the final build recipe records these cache overrides. They leave
the output DLL name and measured installation unchanged.
During the build, its CPU affinity was reduced to four logical CPUs and priority
to below normal after the user's fan-noise report. No performance or memory
measurement runs concurrently with that build.
