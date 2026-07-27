<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: export-onnx
description: >-
  Export a vision/ML model given by a link (HuggingFace repo/URL, GitHub source
  repo, or a checkpoint URL) into an optimized fp16 ONNX model by creating and
  running an Olive recipe under olive-recipes/. Use when the user asks to
  export/convert a model to ONNX, mentions an Olive recipe for a model, or pastes
  a model link with an input shape/dtype for ONNX export.
---

# Export a model to ONNX (Olive recipe)

Create and run an **Olive recipe** that converts a model into an optimized ONNX
model, then verify it. Recipes live under a local clone of `microsoft/olive-recipes`
(cloned into your workspace root, referred to below as `<workspace>`), at
`<workspace>/olive-recipes/<model-slug>/olive/`. Mirror the canonical
`nvidia-resnet50v1.5/olive/` (GitHub CNN) recipe; for a non-square transformer,
mirror `microsoft-swinv2-tiny-patch4-window16-256/olive/`.

NOTE: the team's vision recipes (`nvidia-resnet50v1.5`, `google-mobilenet_v2_1.0_224`,
`qfgaohao-mb1-ssd`, `facebook-detr-resnet-50`, `ultralytics-yolov5lu`,
`microsoft-swinv2-...`) are NOT in the public `microsoft/olive-recipes` — they are
authored locally under the clone. If a referenced recipe is absent, author it from
scratch using this skill (the public repo's `microsoft-resnet-50` is a *different*
HF ResNet-50, not the NVIDIA v1.5 model).

## Inputs (from the user's message)
- **Model link** (required): HuggingFace model id/URL, GitHub repo/path, or a checkpoint URL.
- **Input shape** (optional): e.g. `1x3x224x224`. Default to the model's documented inference size.
- **dtype** (optional): `f16` (default) or `f32`.

Ask a short clarifying question only if the source, input shape, or dtype is
ambiguous AND it materially changes the result (e.g. multiple pretrained variants,
or weights only available behind a Google-Drive folder).

## Scope — is it exportable?
- ✅ Single-image **CNNs and standard transformers**: ResNet, MobileNet, ViT, Swin, CLIP, DETR, SSD, YOLO, etc.
- ❌ **Multi-view 3D-detection stacks** (PETR, BEVFormer, DETR3D): need the `mmdet3d`/`mmcv` ecosystem, multi-input camera geometry, and custom deformable/DCN ops that don't fit this CPU env or standard ONNX. Explain why and stop (offer only a backbone-only CNN stand-in).

## One-time setup (do first if missing)
- **Repos**: clone both into your workspace root `<workspace>` (shallow, to avoid
  the known hang on large assets during a full clone):
  `git clone --depth 1 https://github.com/microsoft/Olive.git` and
  `git clone --depth 1 https://github.com/microsoft/olive-recipes.git`.
- **Env deps**: the `hipdnn-ep` conda env ships `onnx`, `onnxruntime-directml`,
  `transformers`, `onnx_ir` but NOT torch/olive. Install the rest:
  - `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  - `python -m pip install olive-ai onnxscript`
  - Do NOT `pip install onnxruntime` — it collides with the installed
    `onnxruntime-directml`, which already provides `CPUExecutionProvider` (all
    Olive passes here run on CPU). Installing `olive-ai` does NOT pull it in.
  - Env python: the `hipdnn-ep` env's interpreter under your conda/miniforge
    install (find it via `conda env list`; e.g.
    `<miniforge-root>/envs/hipdnn-ep/python.exe` on Windows).
- Verified against **olive-ai 0.13.0** (pass/config names below match this).
  `onnxoptimizer` is optional (peephole logs a benign warning without it).

## Environment & conventions (must follow)
- Run in the `hipdnn-ep` conda env. For models with rich console output (e.g.
  ultralytics), use: `conda run --no-capture-output -n hipdnn-ep python ...`
  and set `PYTHONUTF8=1` (avoids a Windows cp1252 `conda run` crash). Running the
  env python directly (path above) with `$env:PYTHONUTF8=1` also works.
- Scratch downloads/scripts go in a scratch dir under your workspace
  (e.g. `<workspace>/cursor_workspace`), referred to below as `cursor_workspace/`.
- Do NOT `git clone` the **model source** repos (e.g. NVIDIA DeepLearningExamples) —
  they hang on assets. `curl` only the needed model definition file(s) from
  `raw.githubusercontent.com`. (The two Olive repos above are the exception:
  clone them shallow, once.)
- For slow steps (large weight downloads, big exports), run in the background
  (`block_until_ms: 0`), append `&& echo RUN_COMPLETE`, and wait on the output
  pattern (`RUN_COMPLETE`/`Traceback`/`Error`) instead of long fixed timeouts.

## Recipe layout — `olive-recipes/<model-slug>/olive/`
1. **`user_script.py`** — a `model_loader(model_path=None)` returning the `.eval()` PyTorch model with pretrained weights.
   - HuggingFace: load via `transformers`/`diffusers`; wrap so `forward` returns plain tensors (e.g. `out.logits`).
   - GitHub source: vendor the minimal model file(s) into the recipe (as a package if it uses relative imports); shim tiny deps (e.g. a `timm` shim for `to_2tuple`/`DropPath`/`trunc_normal_`); auto-download the checkpoint (torch.hub / URL) in the loader, falling back to random init if unavailable (fine for perf benchmarking — note it).
   - **torchvision shortcut**: when the GitHub model is a standard arch that
     torchvision implements identically (e.g. NVIDIA `resnet50v1.5` == `torchvision.models.resnet50`),
     build from torchvision and load its public ImageNet weights
     (`ResNet50_Weights.IMAGENET1K_V2`) instead of vendoring + fetching an
     authenticated NGC checkpoint. Weights don't change the exported graph or its
     perf; note in the loader which weights (or random init) were used. This is
     what the `nvidia-resnet50v1.5` recipe does.
   - Rebuild at the requested input shape when it differs from default (drop resolution/window-derived buffers, load with `strict=False`).
2. **`config.json`** — input model + `io_config`, passes in order. Exact type
   string is `"PyTorchModel"` (GitHub source) with `"model_script": "user_script.py"`
   and `"model_loader": "model_loader"`, or `"HfModel"`. `io_config` =
   `input_names`, static `input_shapes` = requested shape, `output_names`. Passes:
   - `conversion`: `OnnxConversion` (`target_opset` 17)
   - `peephole`: `OnnxPeepholeOptimizer` (`onnxscript_optimize: true`, `fuse_reshape_operations: true`)
   - `optimize`: `OrtTransformersOptimization` (auto-tuned by `run.py`, see below)
   - `surgery`: `GraphSurgeries` — `"surgeries": [ { "surgeon": "DeduplicateNodes" }, { "surgeon": "InferShapes" } ]`
   - `fp16`: `OnnxFloatToFloat16` (`keep_io_types: false`) — include only for f16
   Define a `systems.local_system` (`"type": "LocalSystem"`,
   `accelerators: [{ "device": "cpu", "execution_providers": [ "CPUExecutionProvider" ] }]`),
   set `"host"`/`"target"` to it, and set `output_dir`/`cache_dir`.
3. **`run.py`** — copy from `nvidia-resnet50v1.5/olive/run.py`. It:
   - resolves `config.json` via `Path(__file__).parent`, loads the source model,
     classifies **CNN vs transformer**, and patches the `optimize` pass;
   - writes `config.generated.json`, calls `olive.workflows.run`, then applies the
     fp16 post-fix (below).
   - **Must run from the recipe's `olive/` dir**: `config.json` paths
     (`model_script`, `output_dir`, `cache_dir`) are relative to cwd. Outputs land at
     `olive/models/<output_dir>/model.onnx`; `cache/` and `config.generated.json`
     are written there too (git-ignore / clean up).

## `model_type` auto-detection (in `run.py`, for `OrtTransformersOptimization`)
Supported ORT set: `bart bert clip conformer gpt2 gpt_neox mmdit phi qwen3 sam2 swin t5 tnlr unet vae vit`.
- **CNN** (Conv-dominant, no attention): `model_type: "bert"`, `opt_level: 1`, ALL `optimization_options` off → ORT basic graph opt only (Conv+BN fold, const fold, DCE).
- **Transformer** (has attention): keep the matched family (`vit`/`swin`/`clip`/…) if set, else default `bert`. For families not in the ORT set (e.g. `detr`), author `opt_level: 1` with fusions off (safe graph opt, no risky mis-fusions).
- HuggingFace source with a local `config.json`: map its `model_type` to the ORT set (see `cursor_workspace/detect_model_type.py` if present, else map by hand).

## fp16 duplicate-node post-fix (required in `run.py`)
fp16 conversion can emit redundant `*_cast_to_fp32` nodes that share a node name
AND redefine the same output tensor, which ORT rejects. After `olive_run`, load each
output `model.onnx` and: drop nodes whose output name is already produced, uniquify
duplicate node names, then **topologically re-sort** the graph. Canonical
implementation: `fix_fp16_model` in `nvidia-resnet50v1.5/olive/run.py` (also in
`ultralytics-yolov5lu`). The fix is a harmless no-op when there are no duplicates,
so include it unconditionally for f16 exports.

## Steps
1. Identify source (HF vs GitHub vs checkpoint URL) and resolve input shape + dtype.
2. Fetch/vendor the model definition + checkpoint; write `user_script.py`.
3. Write `config.json`; copy/adapt `run.py`.
4. `cd` into the recipe's `olive/` dir, then run:
   `conda run --no-capture-output -n hipdnn-ep python run.py 2>&1 && echo RUN_COMPLETE`
   (background + wait on `RUN_COMPLETE`). The export itself is fast (~10-15s once
   weights are cached); the first run also downloads model weights.
5. Verify `olive/models/<output_dir>/model.onnx`: `onnx.checker.check_model`, print
   input/output names/shapes/dtypes (elem_type 10 = float16), and run one
   onnxruntime inference (`providers=['CPUExecutionProvider']`, fp16 input) on random
   input to confirm it loads and yields finite output.
6. Report: recipe path, output model path, detected `model_type`, verified I/O.

## Reference recipes (patterns to mirror; author under the local clone if absent)
- `nvidia-resnet50v1.5` (GitHub CNN) — canonical `run.py`; uses the torchvision
  shortcut (torchvision `resnet50` == v1.5) + `IMAGENET1K_V2` weights, `model_type: bert`,
  `opt_level: 1`, all FusionOptions off. Output `[1,1000]` (add a 1001-class head only
  if parity with an NVIDIA-trained checkpoint is required).
- `microsoft-swinv2-tiny-patch4-window16-256` (transformer, non-square 512x1024).
- `google-mobilenet_v2_1.0_224` (HF CNN via transformers).
- `qfgaohao-mb1-ssd` (vendored package, detector, random-init fallback).
- `facebook-detr-resnet-50` (HF hybrid CNN+transformer; needs `timm`).
- `ultralytics-yolov5lu` (ultralytics; UTF-8/no-capture + fp16 post-fix).
