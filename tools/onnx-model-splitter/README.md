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
| `export_chunk_model.py` | Export fixed-shape Llama prefill/decode ONNX variants with genai_config generation |
| `genai_config_pipeline_from_folder.py` | Generate `decoder-pipeline` style `genai_config.json` for ORT GenAI from an existing model folder |
| `inspect_onnx.py` | Inspect ONNX model structure (inputs, outputs, nodes, op types, preprocessing chains) |
| `verify_ort_inference.py` | Verify extracted models by running ORT inference with dummy inputs (subprocess-isolated) |
| `verify_extraction.py` | Verify extracted model files: sizes, graph structure, initializer integrity, fixed-shape validation |
| `_verify_one_model.py` | Subprocess worker for `verify_ort_inference.py` — loads one model and runs inference |

## Installation

```bash
pip install onnx onnxruntime numpy
pip install onnxoptimizer  # optional but recommended
```

> **Note:** `onnxoptimizer` is optional but recommended. When installed, fixed-shape
> variants automatically run Shape/Gather constant folding to eliminate redundant
> shape-computation nodes.

## Quick Start

### Extract submodels

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

### Export prefill/decode variants

```bash
# Export prefill + decode ONNX with genai configs
python export_chunk_model.py --model path/to/model.onnx --output /output/dir

# Specify max context length
python export_chunk_model.py --model path/to/model.onnx --output /output/dir --max-length 4096
```

### Generate genai_config

```bash
python genai_config_pipeline_from_folder.py path/to/model_folder \
    --fixed-prompt-length 128 --max-length 4096
```

### Inspect a model

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

### Verify extracted models

```bash
# Verify all extracted models via ORT inference
python verify_ort_inference.py /output/dir

# Verify only single_op models
python verify_ort_inference.py /output/dir --mode single_op

# Verify model file structure and fixed-shape correctness
python verify_extraction.py /output/dir

# Skip fixed-shape dimension checks
python verify_extraction.py /output/dir --skip-fixed-shape-check
```

## End-to-End Example: Llama-3.1-8B

Assume the original model is at
`C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml`.

### Step 1 — Extract submodels (single_op / single_layer / full_model)

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

### Step 2 — Export OGA prefill / decode variants

#### 2a. Generate genai_config

```bash
python genai_config_pipeline_from_folder.py ^
    C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml ^
    --max-length 16384 ^
    --fixed-prompt-length 12200
```

This produces `genai_config_pipeline.json` in the model folder, which is
required by the next step.

#### 2b. Export prefill / decode ONNX

```bash
python export_chunk_model.py ^
    --model  C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\model.onnx ^
    --output C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\space\chunk ^
    -T C:\modelzoo\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml\genai_config_pipeline.json
```

Output:

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
