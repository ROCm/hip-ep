<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Dynamic shapes — how they flow through the MorphiZen EP

This page is the 10,000-ft view of dynamic shapes in the MorphiZen EP. It
covers LLM decode/prefill, vision encoders, and anything else with
`-1`/symbolic dims in its ONNX type signature. Point a new contributor here
first; the in-file comments and CLAUDE.md gotchas have the surgical detail.

## The two problems

ONNX models with dynamic shapes give the EP two distinct headaches:

1. **The MLIR compiler needs concrete types** to lower bufferization and
   reshape decomposition. HuggingFace exports often leave intermediate
   tensors as `<?x?x?x?>` even when the surrounding ops uniquely determine
   them, so the compiler has to *infer back* what each op produces.
2. **ORT needs the output shape before it calls `Compute()`**. The EP must
   tell it "this output is `[N, 256, 2560]`" — even when N comes from a
   runtime input dim, and 256 is a value the original model never wrote
   down statically.

Both problems exist for every model, but they wear different masks:

| Model class | Typical loose dims | Where the answer comes from |
|---|---|---|
| **LLM prefill / decode** | `batch`, `sequence_length`, `total_sequence_length` (KV cache) | Shared `dim_param` names: `attention_mask`'s seq dim drives every layer's KV output |
| **Vision encoder** (SigLIP, ViT, …) | `num_images`, intermediate `<?x?x?xH*D>` ViT attention reshapes | Compile-time refinement (most intermediates are derivable) + an SSA trace from `image_features` back to `pixel_values` (the input/output `dim_param`s don't match by name) |
| **Anything weird** | Whatever the exporter left dynamic | One of: a static value the compiler proves, a name-matched input dim, an SSA-traced input dim, or fail loudly |

## Architecture at 10,000 ft

```
┌─────────────────────────────────────────────────────────────────────┐
│  COMPILE                                                            │
│                                                                     │
│   ONNX bytes                                                        │
│       │                                                             │
│       ▼                                                             │
│   InferOnnxShapes ─────► refines op result types in place           │
│       │                  + traces output dims back to input args    │
│       │                  + stashes both for the C ABI               │
│       ▼                                                             │
│   build_metadata_json ─► writes a DimSource per output dim into     │
│       │                  metadata.proto (4-priority resolution)     │
│       ▼                                                             │
│   metadata.proto bytes + model.dll  ─► persisted via EPContext      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │  (disk / cache)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RUNTIME (could be a different process)                             │
│                                                                     │
│   MlirCustomOp::Compute                                             │
│       │                                                             │
│       ▼                                                             │
│   marshal_output_tensors                                            │
│       │   for each output dim:                                      │
│       │     • DimSource says STATIC?  → use the number              │
│       │     • DimSource says LOOKUP?  → read inputs[i].shape[j]     │
│       │     • else                    → fail loudly                 │
│       ▼                                                             │
│   ctx.GetOutput(shape) → ORT allocates → DLL runs inference         │
└─────────────────────────────────────────────────────────────────────┘
```

The persistent channel between compile and runtime is `metadata.proto`.
Everything else (thread-locals, C ABI calls) is internal to the
compile-side process.

## `DimSource`: how the EP picks an output shape

Every dynamic output dim in `metadata.proto` carries one `DimSource`
entry. Three mutually-exclusive states:

```
┌─ STATIC ──────────────────────────────────────────────────────────┐
│  static_value > 0                                                 │
│  Runtime uses the integer directly.                               │
│  Source: graph declared it, OR compiler tightened it.             │
│  Example: Gemma-3 vision output dim 2 = 2560 (declared)           │
│           Gemma-3 vision output dim 1 = 256  (compiler proved)    │
└───────────────────────────────────────────────────────────────────┘

┌─ RUNTIME-INPUT-LOOKUP ────────────────────────────────────────────┐
│  resolved = true, static_value = 0                                │
│  Runtime: round(inputs[input_idx].shape[dim_idx] * mult).         │
│  Source: dim_param name match OR SSA-origin trace.                │
│  mult = 1.0 (identity, default — proto3 0.0 also treated as 1.0). │
│  mult = 1/K when a Reshape in the trace divides dim 0 by K        │
│  (Qwen vision patch merger: mult = 0.25 for divide-by-4).         │
│  mult > 1.0 reserved for future multiply-by-K (upsamplers).       │
│  Examples:                                                        │
│   * Llama present.0.key seq dim ← attention_mask[1] (mult=1.0)    │
│   * Gemma-3 vision dim 0   ← pixel_values[0]        (mult=1.0)    │
│   * Qwen vision dim 0      ← pixel_values[0] * 0.25 (mult=0.25)   │
└───────────────────────────────────────────────────────────────────┘

┌─ UNRESOLVED ──────────────────────────────────────────────────────┐
│  Both above false. Compile-time error.                            │
│  pass_main.cpp::build_metadata_json LOG(FATAL)s with the          │
│  offending output name + dim. Either fix the model (add a         │
│  shared dim_param) or teach InferOnnxShapes a new rule.           │
└───────────────────────────────────────────────────────────────────┘
```

`pass_main.cpp::build_metadata_json` fills the entry in this priority
order (first match wins):

```
       ┌──────────────────────────────┐
       │ 1. STATIC                    │  max(graph_dim, refined_dim)
       │    (compiler-refined         │  — InferOnnxShapes never widens,
       │     OR graph-declared)       │  so max is well-defined.
       └──────────────┬───────────────┘
                      │ no positive value
                      ▼
       ┌──────────────────────────────┐
       │ 2. dim_param NAME MATCH      │  Find an input whose dim_param
       │    (output ↔ input)          │  string equals this output's.
       └──────────────┬───────────────┘
                      │ no match
                      ▼
       ┌──────────────────────────────┐
       │ 3. SSA-ORIGIN TRACE          │  InferOnnxShapes walked SSA
       │    (back through Cast,       │  backward and recorded which
       │     Transpose, MatMul, …)    │  input dim this output dim
       │                              │  ultimately reads from.
       └──────────────┬───────────────┘
                      │ no traceable origin
                      ▼
       ┌──────────────────────────────┐
       │ 4. FAIL LOUDLY               │
       └──────────────────────────────┘
```

## `InferOnnxShapes`: how the compiler tightens what it can

A function-level MLIR pass at the head of `convert-onnx-to-hip`. It does
two jobs in one walk:

```
                ┌──────────────────────────┐
                │  funcOp (onnx.* IR)      │
                └────────────┬─────────────┘
                             │
         ╔═══════════════════╧═══════════════════╗
         ║   1. FORWARD type refinement walk     ║
         ║      (topo order, def-before-use)     ║
         ║                                       ║
         ║   for each onnx.X op:                 ║
         ║     proposal = inferXResultType(...)  ║
         ║     if strictly tighter:              ║
         ║       op.result.setType(proposal)     ║
         ╚═══════════════════╤═══════════════════╝
                             │
         ╔═══════════════════╧═══════════════════╗
         ║   2. BACKWARD SSA-origin trace        ║
         ║                                       ║
         ║   for each function output dim:       ║
         ║     walk SSA back through the same    ║
         ║     op set → (arg_idx, dim_idx)       ║
         ║     or std::nullopt if untraceable    ║
         ╚═══════════════════╤═══════════════════╝
                             │
                             ▼
                ┌──────────────────────────┐
                │  thread-local stash      │
                │  → C ABI → DimSource     │
                └──────────────────────────┘
```

The per-op rules cover the ops every transformer encoder/decoder reaches
for: Reshape, Transpose, MatMul, Cast, unary same-shape ops (Tanh,
Softmax, LayerNorm, …), and binary broadcast ops (Add, Mul, …). The
backward trace adds Conv/Pool/Reduce for batch-dim passthrough.

The pass runs in a round loop with `FastGeluFusion` and
`ProjectorOpsRewrites` because some pre-lowering rewrites (notably the
AvgPool decomposition in the vision projector) emit new ops that need
their own refinement. The loop breaks as soon as a round leaves the IR
untouched — LLMs typically take 2 rounds (mutate + confirm), vision
encoders ~3.

## Three worked examples

### LLM decode (Llama 8B, OGA)

```
 INPUTS                                OUTPUTS
   input_ids        [1,    1]          logits           [1, 1, 128256]
   attention_mask   [1, S+1]    ──┐    present.0.key    [1, 8, S+1, 128]
   past_key_values  [1, 8, S, 128] │    present.0.value  [1, 8, S+1, 128]
   position_ids     [1,    1]    │    …
                                  │
                                  └──► dim_param "total_sequence_length"
                                       links attention_mask[1] to every
                                       present.N.{key,value}[2]
```

DimSource resolution: pure name match (priority 2). `total_sequence_length`
is the dim_param on `attention_mask[1]` AND on every `present.N.{key,value}`
seq dim. At runtime, marshal_output_tensors reads
`inputs[attention_mask].shape[1]` and stamps it into every present output's
seq dim.

OGA's past_present_share_buffer wrinkle: OGA binds the same OrtValue to
both `past_key_values.N.key` (input) and `present.N.key` (output) for
zero-copy KV reuse. `attention_mask` is sized to the tight token count
(e.g. 7), but the past buffer was pre-allocated to max_length (e.g. 128).
The runtime override in `marshal_output_tensors` notices `past_shape >
DimSource_resolved`, takes the past shape instead, and ORT returns the
pre-allocated buffer — pointer identity preserved.

### LLM prefill (same model, longer prompt)

Same DimSource entries; the only difference at runtime is that
`attention_mask[1]` is now the prompt length (e.g. 128). The compiled DLL
is identical — one model.dll serves any prefill length and any decode
position. No recompile per shape.

### Vision encoder (Gemma-3 SigLIP)

```
 INPUTS                                 OUTPUTS
   pixel_values    [num_images, 3,      image_features [num_image_tokens,
                    896, 896]                            MatMulimage_features_dim_1,
                                                         2560]
```

Three output dims, three different resolution mechanisms:

```
  output dim 0 (num_image_tokens):
      name "num_image_tokens" ≠ any input dim_param
      → priority 3: SSA trace from image_features back through
        MatMul → Mul → Div → Sqrt → ... → Reshape → AveragePool
        → Reshape → Transpose → Add → ... → Conv → pixel_values
      → DimSource: {input_idx=pixel_values, dim_idx=0, resolved=true}

  output dim 1 (MatMulimage_features_dim_1):
      name doesn't match an input dim_param
      → InferOnnxShapes tightens it during compile:
        AveragePool 4×4 on [B,1152,64,64] → [B,1152,16,16],
        projector Reshape with -1 cancels to [B, 256, 1152]
      → priority 1: STATIC = 256

  output dim 2 (2560):
      declared static in the graph
      → priority 1: STATIC = 2560
```

No model modification, no in-memory rewrite, no test-framework hooks.
The runtime computes `[N, 256, 2560]` for any `N` the caller passes.

### Vision encoder with a 2x2 patch merger (Qwen3.5 vision)

```
 INPUTS                          OUTPUTS
   pixel_values   [num_patches,    image_features [num_logical_patches,
                   1536]                            4096]
   image_grid_thw [1, 3]
```

`num_patches` and `num_logical_patches` are different strings → name
match (priority 2) fails. The body has a patch-merger Reshape on a
`<num_patches x 1152>` tensor with shape operand `[-1, 4608]`, which
collapses 2x2 spatial windows into the last dim:

  out.dim[0] = in.dim[0] * 1152 / 4608 = in.dim[0] / 4

```
  output dim 0 (num_logical_patches):
      name doesn't match an input dim_param
      → priority 3: SSA trace through the patch-merger Reshape detects
        outOther (4608) > inOther (1152), composes a divisor of 4
        into the trace, continues back to pixel_values[0].
      → DimSource: {input_idx=pixel_values, dim_idx=0, resolved=true,
                    mult=0.25}
      → Runtime: round(pixel_values.shape[0] * 0.25)

  output dim 1 (4096):
      declared static in the graph
      → priority 1: STATIC = 4096
```

`mult` is a `double` field on `DimSource`, bit-cast through the
`int64_t` C ABI buffer. It defaults to `1.0` (proto3 default `0.0` is
remapped to `1.0` at the runtime call site). `mult > 1.0` is reserved
for future multiply-by-K cases (spatial upsamplers) — the trace rule
will refuse to emit it today.

The test that pins this contract is
`test/python/test_qwen3_5_9b.py::test_qwen_vision_patch_merger_dynshape`:
a synthetic ONNX with just the patch-merger Reshape, run on EP at three
input sizes, asserts output dim 0 = num_patches / 4.

## Where things live

| File | What it does |
|---|---|
| `lib/Conversion/OnnxToHip/InferOnnxShapes.cpp` | The pass: forward refinement + backward SSA trace + thread-local stashes |
| `lib/Conversion/OnnxToHip/OnnxResultTypeInference.{h,cpp}` | Pure per-op type rules (also reused by the pre-lowering rewriters) |
| `lib/Conversion/OnnxToHip/OnnxToHip.cpp` | Pre-lowering round loop with quiescence early-exit |
| `lib/CInterface/CompilerAPI.cpp` | C ABI exports (two-call discovery pattern) |
| `backend-mlir-compiler/proto/metadata.proto` | `DimSource` (3-state) |
| `backend-mlir-compiler/level-1-pass/src/pass_main.cpp` | 4-priority DimSource population |
| `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp` | Runtime DimSource resolution + OGA share-buffer override |
| `test/lit/Conversion/onnx-to-hip/test_infer_onnx_shapes.mlir` | Fast regression coverage for each refinement rule |
| `test/python/test_gemma3_4b.py::TestGemma3_4BVisionDynShape` | End-to-end Gemma-3 vision regression suite |
| `test/python/test_qwen3_5_9b.py::test_qwen_vision_patch_merger_dynshape` | Patch-merger Reshape divide-by-K (mult=0.25) end-to-end micro-test |

## When you'll need to touch this

* **A model fails compile with `LOG(FATAL): output dim X is dynamic AND
  has no matching input dim_param AND InferOnnxShapes did not tighten it
  AND SSA-origin trace failed`.** The error message is the recipe: either
  the model is missing a `dim_param` link, OR an op in the trace chain
  isn't in the rule set.

* **A model compiles but the output shape is wrong at runtime.** Check
  the LIT test `test_infer_onnx_shapes.mlir` for the rule you suspect;
  add a case if missing. The canonical regression is MatMul outer-batch
  alignment — see the CLAUDE.md gotcha and the matching LIT case.

* **You're adding a new op to MorphiZen.** If it changes shapes
  non-trivially, add an `inferXResultType` to `OnnxResultTypeInference`
  AND a handler in `InferOnnxShapes::inferOpType` AND (if outputs trace
  through it) a case in `traceDimOrigin`.
