<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: roctx-pipeline-stages
description: Add ROCTx ranges that name the pipeline stage issuing GPU work (embeddings, prefill, decode, lm_head) in an onnxruntime-genai model, and read the resulting per-stage breakdown out of a ktrace capture. Use when a trace shows kernels but not which stage they belong to, when attributing TTFT or per-token cost to stages, or when deciding whether phase attribution belongs in the app, the runtime, or the EP.
---

# Naming the pipeline stage in a ktrace capture

A capture tells you which kernel ran and, through the EP's own instrumentation,
which operator it belonged to. It does not tell you **which pipeline stage
issued it**. For a genai pipeline model those stages are separate sessions over
separate graphs, and in the timeline they are separated only by things you have
to guess at: the shapes their kernels happen to use, or the gaps between them.

Guessing is expensive. On one investigation it cost several rounds of chasing a
"1.15 ms stream-sync wakeup" that did not exist, because the window boundaries
had been drawn by hand from kernel shapes and were 3 ms short. The authoritative
boundary was one range away.

## Three layers, and which one is missing

| layer | emitted by | granularity |
|---|---|---|
| `oga.prefill`, `oga.decode`, `oga.e2e` | the benchmark app | whole phase |
| `oga.stage.<name>` | **the genai library** | one session Run |
| `qgemm`, `qadd`, `rms_norm` | the EP | one operator |

The app layer brackets the phase. The EP layer names the operator. Without the
middle layer you cannot say which of four sessions a given `qgemm` came from,
and stages that run in both phases (embeddings, lm_head) are indistinguishable.

## Where the stage name comes from

Not from inference. `genai_config.json` already names each stage:

```json
"pipeline": [
  { "embeddings": { "filename": "..._emb.onnx" } },
  { "prefill":    { "filename": "..._128_no_taps.onnx" } },
  { "decode":     { "filename": "..._1_no_taps.onnx" } },
  { "lm_head":    { "filename": "..._lm_head.onnx" } }
]
```

`IntermediatePipelineState` already holds that name as
`config_->model.decoder.pipeline[id_].model_id`. Use it verbatim.

## The change

Three files in the genai source tree.

**`src/roctx_range.h`** (new) — a header-only, env-gated, lazily-loaded wrapper.
Mirror whatever the benchmark's `main.cpp` already does so there is one idiom,
not two. The important detail is that constructing a range with an **empty
name is a no-op**, so the hot path costs one empty-string test when tracing is
off:

```cpp
class RoctxRange {
 public:
  explicit RoctxRange(const std::string& name)
      : pushed_{!name.empty() && RoctxApi::Get().Push(name.c_str()) >= 0} {}
  ~RoctxRange() { if (pushed_) RoctxApi::Get().Pop(); }
};
```

`RoctxApi` loads `roctx64.dll` only when `HIPDNN_EP_ROCTX=1`, resolving
`roctxRangePushA` and `roctxRangePop`.

**`src/models/decoder_only_pipeline.h`** — one member on
`IntermediatePipelineState`:

```cpp
std::string roctx_label_;  // "oga.stage.<model_id>", empty when tracing is off
```

**`src/models/decoder_only_pipeline.cpp`** — build the label once in the
constructor, wrap the session run:

```cpp
// in the constructor body
if (RoctxApi::Get().IsAvailable())
  roctx_label_ = "oga.stage." + model_.config_->model.decoder.pipeline[id_].model_id;

// in IntermediatePipelineState::Run, around State::Run
{
  RoctxRange stage_range{roctx_label_};
  State::Run(*model_.sessions_[id_]);
}
```

Build it the way CI does (`build.py --config Release --cmake_generator Ninja
--use_dml --ort_home ... --skip_tests --skip_examples`) and deploy the resulting
`onnxruntime-genai.dll`. `--skip_tests` does not stop `unit_tests.exe` from
being built and it may fail to link; the library and `model_benchmark.exe` are
produced regardless, so check for those rather than the build's exit code.

## Why not put this in the EP

Tried, and it does not work for this model:

- `graph_viewer.name()` is available to the EP but returns **empty** — the
  exported graphs carry no name attribute.
- `genai_config.json` has no per-stage `session_options`, so a label cannot be
  passed down without first extending the genai library anyway.
- The EP could emit an opaque per-session id (the `RuntimeState*`), which works
  but needs a manual id-to-stage mapping on every analysis.

And a label the EP *infers* will eventually lie. Inferring "decode" from `M == 1`
holds today and breaks the moment speculative decoding runs — a model directory
may already ship a `..._6_spec_dec.onnx` with `M == 6`, which is still decode.
Take the name from configuration, not from shape.

## Reading it back

ROCTx push/pop is **stack-based per thread**. `Range_Id` in
`trace_roctx_trace.csv` is the nesting level, not a unique pair id, so pair by
stack:

```python
stacks = collections.defaultdict(list); spans = []
for x in sorted(rows, key=lambda r: int(r['Timestamp'])):
    tid, ts = x['Thread_Id'], int(x['Timestamp'])
    if x['Operation'] == 'RANGE_PUSH':
        stacks[tid].append((x['Message'], ts))
    elif stacks[tid]:
        m, t0 = stacks[tid].pop()
        spans.append((t0, ts, m))
```

Then sum each `oga.stage.*` inside the enclosing `oga.prefill` / `oga.decode`
span. What is left over is the cost of moving between stages -- session
switches, the host work between them -- which is otherwise invisible.

A real result, for scale:

```
oga.prefill  90.02 ms          oga.decode  21.54 ms/token
  embeddings    1.11   1.2%      embeddings   0.499   2.3%
  decoder      76.19  84.6%      decoder     18.382  85.4%
  lm_head       6.55   7.3%      lm_head      1.423   6.6%
  between       6.18   6.9%      between      1.233   5.7%
```

## Traps

**Markers that fire once get lost.** `oga.prefill` appears once per run; a
`Counter(...).most_common()` view buries it under a hundred thousand operator
ranges. When first opening a ROCTx trace, print the **set of distinct messages**,
not the frequency ranking.

**Do not derive phase windows from kernel features.** "First WMMA dispatch to
the last lm_head kernel" looks reasonable and silently omits the head and tail
of the phase. Cross-check any hand-drawn window against the range that already
brackets it.

**The app-level and stage-level ranges nest.** `oga.decode` contains 127
`oga.stage.decode` ranges, not one. Sum them; do not assume a one-to-one match.
