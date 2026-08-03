<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Export a model to ONNX

Export the model I provide into an optimized ONNX model by authoring and running an
Olive recipe, following the **export-onnx** skill end to end:
@.cursor/skills/export-onnx/SKILL.md

## Inputs (parse from the text after the command; ask only if genuinely ambiguous)
- **Model link** (required): HuggingFace id/URL, GitHub repo/path, or a checkpoint URL.
- **Input shape** (optional): e.g. `1x3x224x224`. Default to the model's documented inference size.
- **dtype** (optional): `f16` (default) or `f32`.

If no model link is present in my message, ask me for one before doing anything else.

## What to do
1. Confirm the source is in scope (single-image CNN or standard transformer per the skill).
2. Do the one-time setup only if missing (Olive repos cloned, `hipdnn-ep` env deps).
3. Author the recipe under `olive-recipes/<model-slug>/olive/` (`user_script.py`,
   `config.json`, `run.py`), mirroring the canonical reference recipes.
4. `cd` into the recipe's `olive/` dir and run `run.py` (background + wait on
   `RUN_COMPLETE`); apply the fp16 duplicate-node post-fix for f16 exports.
5. Verify the output `model.onnx` (checker, I/O names/shapes/dtypes, one ORT inference).
6. Report: recipe path, output model path, detected `model_type`, and verified I/O.

Follow the skill's exact pass/config names, environment conventions, and gotchas
(especially running from the recipe's `olive/` dir).
