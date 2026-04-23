# ONNX Model Splitter

A set of tools for splitting ONNX models into submodels for performance testing
and analysis. Supports LLM (text), Vision, and Embedding models.

## Overview

| Script | Description |
|--------|-------------|
| `extract_submodels.py` | Main extraction tool — splits an ONNX model into `single_op`, `single_layer`, and `full_model` submodels |
| `inspect_onnx.py` | Inspect ONNX model structure (inputs, outputs, nodes, op types) |
| `verify_ort_inference.py` | Verify extracted models by running ORT inference with dummy inputs |
| `verify_extraction.py` | Verify extracted model files: sizes, graph structure, initializer integrity |

## Installation

```bash
pip install -r requirements.txt
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
```

### Verify extracted models

```bash
# Verify all extracted models via ORT inference
python verify_ort_inference.py /output/dir

# Verify only single_op models
python verify_ort_inference.py /output/dir --mode single_op

# Verify model file structure
python verify_extraction.py /output/dir
```

## Supported Model Types

The extractor auto-detects the model type based on graph input names:

| Model Type | Detection | Dimension Variants |
|------------|-----------|-------------------|
| **LLM** (text) | `inputs_embeds` or `input_ids` without `image_features` | `sequence_length`: 1, 128, 256, 512, 1024, 2048, 3072 |
| **Vision** | `pixel_values` | `num_patches`: 1024, 1200, 2520, 3600, 4096, 8160 |
| **Embedding** | `input_ids` + `image_features` | `sequence_length` × `num_logical_patches` combinations |

## Output Structure

```
output_dir/
├── single_op/           # One submodel per unique op type
│   ├── MatMul/
│   │   ├── MatMul_dynamic.onnx
│   │   ├── MatMul_seq1.onnx
│   │   ├── MatMul_seq128.onnx
│   │   └── ...
│   ├── Gemm_linear/
│   └── ...
├── single_layer/        # One transformer layer + pre/post processing
│   ├── single_layer_dynamic.onnx
│   ├── single_layer_seq1.onnx
│   ├── ...
│   └── weights.data     # Shared weights (first variant only)
└── full_model/          # Full model with fixed shapes
    ├── full_model_dynamic.onnx
    ├── full_model_seq1.onnx
    ├── ...
    └── weights.data
```

## Extraction Details

### single_op

Extracts one representative node per unique operation type. For LLM models,
ops within repeated layers are deduplicated by type (e.g., one `MatMul` for all
layers). Ops with shape-dependent behavior (e.g., `MatMulNBits` with different
weight sizes) produce separate submodels.

For vision/embedding (flat) models, performance-significant ops (`Gemm`,
`MatMul`, `Conv`, `LayerNormalization`, `Loop`, etc.) retain shape-based
variants, while simple ops (`Add`, `Mul`, `Cast`, etc.) are deduplicated to a
single representative.

### single_layer

- **LLM**: Pre-layer constants + layer 0 + final norm + LM head
- **Vision**: Pre-block processing (Conv3d, position encoding) + ViT block 0 +
  post-block processing (final norm, merger FC, token merging). Equivalent to
  running the full model with only 1 iteration of the repeated blocks.
- **Embedding**: Not applicable (no layer structure)

### full_model

- **LLM**: Full model with redundant branches removed, fixed sequence lengths
- **Vision / Embedding**: Full model with fixed dimension variants (no node
  removal)

### Graph Optimizations

For fixed-shape variants (all except `dynamic`), the following optimizations
are applied automatically when `onnxoptimizer` is installed:

- **eliminate_shape_gather**: Folds `Shape → Gather` chains into constants
  when tensor shapes are fully known
- **eliminate_shape_op**: Removes `Shape` nodes whose outputs are unused
  after gather elimination
- **extract_constant_to_initializer**: Converts `Constant` nodes to
  initializers
- **eliminate_deadend / eliminate_unused_initializer**: Cleans up orphaned
  nodes and weights

This eliminates runtime shape-computation overhead (e.g., mrope Shape→Gather→
Mul→Range chains in Attention layers).