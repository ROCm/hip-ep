<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP shape contract inventory

This ledger is the reviewed classification for every HIP DPS op. `test/lit/Dialect/hip-shape-contract-inventory.mlir` cross-checks the op and contract columns against TableGen. Status entries are migration work, not separate contract categories. Every leaf must inherit exactly one explicit contract base: `Hip_DpsOp_SameShape`, `Hip_DpsOp_Broadcast`, `Hip_DpsOp_Reduction`, `Hip_DpsOp_Semantic`, `Hip_DpsOp_Payload`, or `Hip_DpsOp_OutsAuthoritative`. The default local build runs the audit and fails when a DPS leaf uses `Hip_DpsOp` directly, mismatches the inventory, or bypasses the required verifier/reify policy.

| HIP op | Contract | Shape rule / payload policy | Status |
|---|---|---|---|
| `abs` | `same_shape` | Result shape equals `input` | shared named-source base |
| `add` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `and` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `bias_gelu` | `same_shape` | Result shape equals `data` | shared named-source base |
| `cast` | `same_shape` | Result shape equals `input` | shared named-source base |
| `causal_conv_with_state` | `semantic` | Runtime-supported 1D: output=input; state `[B,C,weight.K-1]` | shared infer/reify/verifier |
| `ceil` | `same_shape` | Result shape equals `input` | shared named-source base |
| `compress` | `payload` | Selected count comes from condition payload; outs/reify policy | audited payload policy |
| `conv` | `semantic` | N from input, C from weights, ONNX spatial convolution formula | shared infer/reify/verifier |
| `conv_transpose` | `semantic` | N from input, C=weights[1]*group, ONNX transpose formula | shared |
| `cos` | `same_shape` | Result shape equals `input` | shared named-source base |
| `cumsum` | `same_shape` | Result shape equals `x` | shared named-source base |
| `div` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `equal` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `exp` | `same_shape` | Result shape equals `input` | shared named-source base |
| `expand` | `payload` | Target shape tensor; constant fold or synchronized readback | audited payload policy |
| `fast_gelu` | `same_shape` | Result shape equals `input` | shared named-source base |
| `gather` | `semantic` | data prefix + indices shape + data suffix | shared infer/reify/verifier |
| `gather_block_quantized` | `semantic` | Gather over logical data; byte-packed int4 doubles the surviving quantize-axis extent | shared infer/reify/verifier |
| `gather_elements` | `same_shape` | Result shape equals `indices` | shared named-source base |
| `gather_nd` | `semantic` | data batch prefix + indices outer dims + data tail | shared infer/reify/verifier; dynamic tuple width uses outs fallback |
| `gelu` | `same_shape` | Result shape equals `input` | shared named-source base |
| `gemm` | `semantic` | Transpose-aware M/N from A/B; C validates only | shared |
| `global_pool` | `semantic` | N/C from input; every spatial result extent is 1 | shared infer/reify/verifier |
| `gqa` | `semantic` | Output follows query; present BNSH capacity is `max(matching past dim 2, total_seq_len)` (or logical length alone without past); optional QK stays `[B,H,Sq,logical Skv]` | shared converter extent utility + payload fallback + partial static verifier |
| `hipdnn_graph` | `outs_authoritative` | Outlined hipDNN graph owns output metadata | audited outs-authoritative |
| `layer_norm` | `semantic` | Y=input shape; Mean/InvStdDev=keepdims reduction at axis | shared |
| `leaky_relu` | `same_shape` | Result shape equals `x` | shared named-source base |
| `less` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `linear_attention` | `semantic` | Output `[B,T,max(Hq,Hkv)*Dv]`; state `[B,Hkv,Dk,Dv]` | shared infer/reify/verifier |
| `log` | `same_shape` | Result shape equals `input` | shared named-source base |
| `matmul` | `semantic` | Broadcasted batch + M from A + N from B | shared |
| `matmul_nbits` | `semantic` | A leading dims plus static N attribute | shared infer/reify/verifier |
| `max` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `min` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `miopen.add` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `miopen.softmax` | `same_shape` | Result shape equals `input` | shared named-source base |
| `mod` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `mul` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `multi_head_attention` | `semantic` | Default runtime: separate rank-3 fp16 Q/K/V; output exactly query shape; no optional inputs/outputs | shared infer/reify/verifier; GQA forms routed before default |
| `neg` | `same_shape` | Result shape equals `input` | shared named-source base |
| `nonzero` | `payload` | Capacity from input numel; true count produced at runtime | audited payload policy |
| `not` | `same_shape` | Result shape equals `input` | shared named-source base |
| `one_hot` | `payload` | Inserted axis extent comes from depth payload readback | audited payload policy |
| `or` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `pad` | `payload` | Constant/stamped pads use exact affine input extents; runtime pads use synchronized converter readback and outs reify | affine refinement complete |
| `pool` | `semantic` | N/C from input plus ONNX pooling spatial formula; optional indices match values | shared infer/reify/verifier |
| `qmoe` | `same_shape` | Result shape equals `input` | shared named-source base |
| `range` | `payload` | Length depends on start/limit/delta payload values | audited payload policy |
| `reciprocal` | `same_shape` | Result shape equals `input` | shared named-source base |
| `reduce_l2` | `reduction` | ONNX axes/keepdims/noop reduction shape | shared |
| `reduce_max` | `reduction` | ONNX axes/keepdims/noop reduction shape | shared |
| `reduce_mean` | `reduction` | ONNX axes/keepdims/noop reduction shape | shared |
| `reduce_min` | `reduction` | ONNX axes/keepdims/noop reduction shape | shared |
| `reduce_prod` | `reduction` | ONNX axes/keepdims/noop reduction shape | shared |
| `reduce_sum` | `reduction` | ONNX axes/keepdims/noop reduction shape | shared |
| `resize` | `semantic` | N/C from input; static spatial extents from imported output template because sizes/scales are not carried | shared infer/reify/verifier |
| `rms_norm` | `same_shape` | Result shape equals `input` | shared named-source base |
| `rope` | `same_shape` | Result shape equals `input` | shared named-source base |
| `scatter_elements` | `same_shape` | Result shape equals `data` | shared named-source base |
| `scatter_nd` | `same_shape` | Result shape equals `data` | shared named-source base |
| `sigmoid` | `same_shape` | Result shape equals `input` | shared named-source base |
| `sign` | `same_shape` | Result shape equals `input` | shared named-source base |
| `silu` | `same_shape` | Result shape equals `input` | shared named-source base |
| `sin` | `same_shape` | Result shape equals `input` | shared named-source base |
| `size` | `semantic` | Rank-zero scalar result | shared infer/converter/verifier |
| `skip_rms_norm` | `semantic` | Output and optional residual equal input; training stats unsupported | shared infer/reify/verifier |
| `slice` | `payload` | Exact for constant parameters with static sliced-axis bounds; dynamic clamps/runtime parameters keep physical capacity + outs policy | complete; dynamic clamp deliberately capacity-authoritative |
| `softplus` | `same_shape` | Result shape equals `input` | shared named-source base |
| `sqrt` | `same_shape` | Result shape equals `input` | shared named-source base |
| `sub` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |
| `tanh` | `same_shape` | Result shape equals `input` | shared named-source base |
| `tile` | `payload` | Each extent=inputDim*repeats payload | refinement complete; shared constant rule + one bulk runtime readback |
| `top_k` | `payload` | Selected axis extent comes from K payload | audited payload policy |
| `transpose` | `semantic` | output[i] = input[perm[i]] | shared infer/reify/verifier |
| `where` | `broadcast` | NumPy right-aligned broadcast over declared inputs | shared |

## Contract meanings

- `same_shape`: result shape equals one named input; converter/reify/verifier use that source.
- `broadcast`: shared NumPy broadcast rule.
- `reduction`: shared ONNX axes/keepdims rule.
- `semantic`: dedicated operand/attribute formula.
- `payload`: output shape depends on tensor payload values and uses an explicit constant-fold/readback/runtime policy.
- `outs_authoritative`: the converter/runtime graph owns correct output metadata and reify intentionally lifts outs.

Every row must inherit the exact explicit base for its contract. The three
mechanical bases generate their shared shape behavior; the semantic, payload,
and outs-authoritative wrappers only preserve the leaf's existing options while
making its reviewed policy explicit. `Hip_DpsOp_SameShape` additionally records
the named source accessor and generates reification plus structural DPS and
static output verification. No DPS leaf may inherit `Hip_DpsOp` directly or
retain `unclassified` in TableGen.

Every `broadcast` and `reduction` row inherits a generated verifier from its
parameterized TableGen base. Those verifiers compose the full structural DPS
contract (uniform tensor/memref mode and tensor result/init parity) before
checking shape semantics. Broadcast verifies all statically decidable extents.
Reduction verifies exact output shape only for compile-time axes; payload-
dynamic axes retain the converter-selected outs shape after rank, dtype, and
attribute validation.
