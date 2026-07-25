<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ONNX Model Splitter

A set of tools for splitting ONNX models into submodels for performance testing
and analysis. Supports LLM (text), Vision, and Embedding models.

## Overview

| Script | Description |
|--------|-------------|
| `extract_submodels.py` | Main extraction tool — splits an ONNX model into `single_op`, `single_layer`, and `full_model` submodels with multiple dimension variants |
| `export_chunk_model.py` | Export fixed-shape Llama prefill/decode ONNX variants with genai_config generation. **Default:** one sliding `p512…` prefill/decode pair only; pass **`--all`** for the full sequence-length matrix (`DEFAULT_SEQ_LENS` in code) and derived fixed-prompt / control stems |
| `compare_chunk_export_outputs.py` | Validate an `export_chunk_model.py` output directory: structural match of each prefill/decode ONNX pair plus `genai_config_*.json` filenames and pipeline I/O lists vs graphs |
| `genai_config_pipeline_from_folder.py` | Generate `decoder-pipeline` style `genai_config.json` for ORT GenAI from an existing model folder |
| `inspect_onnx.py` | Inspect ONNX model structure (inputs, outputs, nodes, op types, preprocessing chains) |
| `verify_ort_inference.py` | Verify extracted models by running ORT inference with dummy inputs (subprocess-isolated) |
| `verify_extraction.py` | Verify extracted model files: sizes, graph structure, initializer integrity; fixed-shape **phase-1** scan and optional **phase-2** `shape_inference` + `check_model` (unless `--skip-shape-inference-check`). Use `--skip-fixed-shape-check` to skip fixed-shape checks entirely |
| `_verify_one_model.py` | Subprocess worker for `verify_ort_inference.py` — loads one model and runs inference |

## Installation

**Recommended:** use an isolated **Conda** environment so `onnx` / `onnxruntime` versions stay consistent with other projects on the same machine.

```bash
conda create -n onnx-splitter python=3.11 -y
conda activate onnx-splitter
```

Then install dependencies (same packages if you use `venv` + `pip` instead):

```bash
pip install -r requirements.txt
```

## Quick Start

End-to-end flow on one machine: **extract** submodels into `space`, **verify** them, build **decoder-pipeline** `genai_config`, **export** prefill/decode ONNX into `space\chunk` (the walkthrough uses **`--all`** so `chunk/` lists every variant; omit it for the default single `p512…` sliding pair only), then optionally **compare** exports. Uses a concrete **Llama-3.1-8B** Windows tree (`^` line continuation for `cmd`); adapt paths to your layout.

### Walkthrough: Llama-3.1-8B (Windows)

Assume the original model is at
`C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml`.

### Step 1 — Extract submodels into `space`

#### 1a. Run extraction (single_op / single_layer / full_model)

```bash
python extract_submodels.py ^
    --model  C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\model.onnx ^
    --output C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\space
```

Output:

```
space/
├── single_op/           # one submodel per unique op type
├── single_layer/        # layer 0 + pre/post processing
└── full_model/          # full model with dimension variants
```

#### 1b. (Optional) Verify `space` (structure + ORT smoke)

Run from the directory that contains the scripts (or use full paths to `verify_extraction.py` / `verify_ort_inference.py`). Both commands should exit with code **0**.

```bash
python verify_extraction.py ^
    C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\space

python verify_ort_inference.py ^
    C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\space
```

`verify_extraction.py` checks extracted `.onnx` layout, external data offsets, graph sanity, and (for fixed-shape filenames) a two-phase shape check: static dims first, then optional ONNX `shape_inference` / `check_model` when phase-1 passes. `verify_ort_inference.py` loads each extracted model in a **subprocess** and runs ORT with dummy inputs. Add `--skip-fixed-shape-check` to skip fixed-shape checks, or `--skip-shape-inference-check` to keep phase-1 but skip phase-2.

### Step 2 — Export OGA prefill / decode variants

#### 2a. Generate genai_config

```bash
python genai_config_pipeline_from_folder.py ^
    C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml ^
    --max-length 16384 ^
    --fixed-prompt-length 12200
```

This produces `genai_config_pipeline.json` in the model folder, which is
required by the next step. When prefill/decode ONNX files are discoverable under
the model directory (or common subdirs such as `chunk/`), pipeline **input/output
name lists** are taken from the graphs so bindings match the export (e.g. sparse KV
or extra state tensors). Use `--no-onnx-io` to force template-only lists; `--onnx-subdir`
prepends extra search paths relative to `model_dir`.

#### 2b. Export prefill / decode ONNX

```bash
python export_chunk_model.py ^
    --model  C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\model.onnx ^
    --output C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\space\chunk ^
    -T C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\genai_config_pipeline.json ^
    --all
```

`--all` exports the **full** variant matrix (multiple `prefill_*` / `decode_*` / `genai_config_*` files). Omit `--all` for a **quick** export: a single default `p512…` prefill/decode pair and matching `genai_config` (plus shared weights).

Output (with `--all`):

```
space/chunk/
├── prefill_p128m16384.onnx
├── decode_p128m16384.onnx
├── prefill_p2048m16384.onnx
├── decode_p2048m16384.onnx
├── prefill_12200.onnx
├── decode_12200.onnx
├── genai_config_p128m16384.json
├── genai_config_p2048m16384.json
├── genai_config_12200.json
├── ...
└── model.onnx.data          # shared external weights
```

#### 2c. (Optional) Compare prefill / decode / genai artifacts

```bash
python compare_chunk_export_outputs.py ^
    C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\space\chunk
```

## Script CLI reference

Copy-paste examples for **each** Python entry point using generic paths. For a full pipeline in order, see **Quick Start** above.

### `extract_submodels.py`

```bash
# Extract all (single_op + single_layer + full_model)
python extract_submodels.py --model path/to/model.onnx

# Extract to a custom output directory
python extract_submodels.py --model path/to/model.onnx --output /output/dir

# Extract only single_op
python extract_submodels.py --model path/to/model.onnx --only single_op

# Extract only single_layer
python extract_submodels.py --model path/to/model.onnx --only single_layer

# Extract only full_model
python extract_submodels.py --model path/to/model.onnx --only full_model
```

### `export_chunk_model.py`

Required: `--model`, `-o` / `--output`. With genai JSON (default), also pass **`-T` / `--config-template`** unless `--no-genai-config`.

```bash
# Quick export: single default sliding pair + genai_config (needs -T)
python export_chunk_model.py --model path/to/model.onnx -o /output/dir -T path/to/genai_template.json

# Full variant matrix (same stems as a “full” split export)
python export_chunk_model.py --model path/to/model.onnx -o /output/dir -T path/to/genai_template.json --all
```

### `compare_chunk_export_outputs.py`

After `export_chunk_model.py`, run on the **same output directory** to check that each `prefill_TAG.onnx` has a matching `decode_TAG.onnx` and `genai_config_TAG.json` (non-`*_dml`): same graph topology (node names, op types, edges), compatible dtypes (intermediate shapes may differ by design), aligned graph I/O except known prefill vs decode sequence axes, and consistent `past_*` / `present_*` shapes. Optional `--compare-intermediate-shapes` enforces identical tensor shapes on every edge.

```bash
python compare_chunk_export_outputs.py /path/to/export_chunk_model_output

python compare_chunk_export_outputs.py /path/to/out --quiet

python compare_chunk_export_outputs.py /path/to/out --compare-intermediate-shapes
```

### `genai_config_pipeline_from_folder.py`

```bash
python genai_config_pipeline_from_folder.py path/to/model_folder \
    --fixed-prompt-length 128 --max-length 4096
```

### `inspect_onnx.py`

```bash
# Basic info (inputs, outputs, summary)
python inspect_onnx.py model.onnx

# List all nodes
python inspect_onnx.py model.onnx --nodes

# Filter by op type
python inspect_onnx.py model.onnx --nodes --type-name MatMul

# Filter by layer index
python inspect_onnx.py model.onnx --nodes --layer 0

# Show op type statistics
python inspect_onnx.py model.onnx --op-types

# Find preprocessing chains (attention_mask, position_ids)
python inspect_onnx.py model.onnx --find-chains
```

### `verify_ort_inference.py` / `verify_extraction.py`

```bash
# Verify all extracted models via ORT inference
python verify_ort_inference.py /output/dir

# Verify only single_op models
python verify_ort_inference.py /output/dir --mode single_op

# Verify model file structure and fixed-shape correctness
python verify_extraction.py /output/dir

# Skip fixed-shape dimension checks (phase-1 + phase-2 off for shapes)
python verify_extraction.py /output/dir --skip-fixed-shape-check

# Keep phase-1 fixed-shape scan; skip ONNX shape_inference / check_model (phase-2)
python verify_extraction.py /output/dir --skip-shape-inference-check
```

## Supported Model Types

The extractor auto-detects the model type based on graph input names:

| Model Type | Detection Rule | Dimension Variants |
|------------|----------------|-------------------|
| **LLM** (text) | Default (has `input_ids` or `inputs_embeds`) | `sequence_length`: 1, 128, 256, 512, 1024, 2048, 3072 |
| **Vision** | Has `pixel_values` input | `num_patches`: 1024, 1200, 2520, 3600, 4096, 8160 |
| **Embedding** | Has both `input_ids` and `image_features` | `sequence_length` x `num_logical_patches` (13 combinations) |

Each model type produces **1 dynamic + N fixed** variants. The dynamic variant
preserves symbolic dimension names; fixed variants replace them with concrete values.

## Output Structure

```
output_dir/
├── single_op/              # One submodel per unique op type
│   ├── MatMul/
│   │   ├── MatMul_dynamic.onnx
│   │   ├── MatMul_seq1.onnx
│   │   ├── MatMul_seq128.onnx
│   │   └── ...
│   ├── GroupQueryAttention/
│   └── ...
├── single_layer/           # One transformer/ViT layer + pre/post processing
│   ├── single_layer_dynamic.onnx
│   ├── single_layer_seq1.onnx
│   ├── ...
│   └── <model>.onnx.data   # Shared external weights
└── full_model/             # Full model with fixed shapes
    ├── full_model_dynamic.onnx
    ├── full_model_seq1.onnx
    ├── ...
    └── <model>.onnx.data   # Copied from original (not rewritten)
```

> **External data naming**: The output `.data` filename matches the original model's
> external data filename (e.g., `model.onnx.data`, `gemma-3-vision.onnx.data`).

## Extraction Details

### single_op

Extracts one representative node per unique operation type.

- **LLM models**: Ops within repeated layers are deduplicated — one per op type.
  Non-layer ops with shape-dependent behavior (e.g., `MatMulNBits` with different
  weight sizes) produce separate submodels.
- **Vision / Embedding (flat) models**: Performance-significant ops (`Gemm`,
  `MatMul`, `MatMulNBits`, `Conv`, `GroupQueryAttention`, `QMoE`,
  `LayerNormalization`, etc.) retain shape-based variants, while simple ops
  (`Add`, `Mul`, `Cast`, `Gather`, etc.) are deduplicated to a single representative.

### single_layer

- **LLM**: Pre-layer constants + layer 0 + final norm + LM head.
- **Vision**: Pre-block processing (Conv3d, position encoding) + ViT block 0 +
  post-block processing (final norm, merger FC, token merging). Equivalent to
  running the full model with only 1 iteration of the repeated blocks.
- **Embedding**: Not applicable (no layer structure).

### full_model

- **LLM**: Full model with redundant preprocessing branches removed
  (`attention_mask → Shape → Gather → Cast` chain,
  `input_ids → Shape → Gather → Unsqueeze → Concat → Reshape` chain).
  The dynamic variant exposes `total_sequence_length` as a graph input for GQA;
  fixed variants use a constant initializer.
  External weight data is **copied** from the original (not re-serialized),
  preserving exact binary identity and avoiding rewriting large `.data` files.
- **Vision / Embedding**: Full model with fixed dimension variants (no node removal).

### Config-based Dimensions

When a `genai_config.json` exists alongside the model, the extractor automatically
reads `num_attention_heads`, `num_key_value_heads`, `head_size`, and `hidden_size`
from `model.decoder` and applies them as fixed dimension substitutions. This
resolves symbolic dims like `num_attention_heads` in Reshape nodes without
requiring manual configuration.

### Graph Optimizations

For fixed-shape variants (all except `dynamic`), the following optimizations
are applied automatically when `onnxoptimizer` is installed:

- **Shape inference**: `onnx.shape_inference.infer_shapes()` resolves intermediate
  symbolic dimensions (e.g., `u1`, `u2`) from known input shapes.
- **eliminate_shape_gather**: Folds `Shape → Gather` chains into constants
  when tensor shapes are fully known.
- **eliminate_shape_op**: Removes `Shape` nodes whose outputs are unused
  after gather elimination.
- **extract_constant_to_initializer**: Converts inline `Constant` nodes to
  initializers.
- **eliminate_deadend / eliminate_unused_initializer**: Cleans up orphaned
  nodes and weights.

### Orphan Removal

Before saving, the extractor automatically detects and removes:

- **Orphan graph inputs**: Inputs declared in the graph but not consumed by any node.
- **Orphan nodes**: Nodes whose outputs are not consumed by any downstream node
  or graph output (iteratively removed until stable).
