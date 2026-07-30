<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Split-Pipeline to Merged ONNX Converter

`run_merged_convert.py` is the CLI entry point. It converts a **split inference pipeline** (embedding, decoder prefill/decode, language-model head) into a **single merged ONNX model** suitable for ONNX Runtime GenAI.

Implementation is split across the `merged_convert/` package (see [Code layout](#code-layout) below).

## Requirements

- Python 3.10+
- Packages: `onnx`, `numpy`

```bash
pip install onnx numpy
```

## Usage

```bash
python run_merged_convert.py \
  --input-dir  path/to/model-bundle \
  --output-dir path/to/output
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--input-dir` | Yes | Directory containing the split ONNX files (see below). |
| `--output-dir` | No | Output directory. Default: `<input-dir>/merged` |

### Example

```bash
python run_merged_convert.py \
  --input-dir ./my-model-bundle \
  --output-dir ./my-model-bundle/merged
```

After conversion, copy tokenizer files into the output directory if you plan to run inference or benchmarks from that folder:

```bash
cp -r my-model-bundle/tokenizer/* my-model-bundle/merged/
```

## Input directory layout

The converter expects **four ONNX files** in the same directory (or only the files listed below—extra ONNX files are ignored unless they form another valid decoder pair).

### Required files

| Role | Filename pattern | Notes |
|------|------------------|-------|
| Decoder (prefill) | `*_128.onnx` **or** `*_0.onnx` | Fixed prefill sequence length (typically 128). |
| Decoder (decode) | Matching `*_1.onnx` | Single-token decode step; stem must match the prefill file (e.g. `model_128.onnx` + `model_1.onnx`). |
| Embedding | `*_emb.onnx`, `*embedding*.onnx`, or `*embeddings*.onnx` | First match wins. |
| Language-model head | `*_lm_head.onnx` or `*lm_head*.onnx` | First match wins. |

**Example (minimal bundle):**

```text
model-bundle/
├── model_128.onnx      # decoder prefill
├── model_1.onnx        # decoder decode
├── model_emb.onnx      # embedding
├── model_lm_head.onnx  # lm head
└── model.data          # external weights (if used)
```

### Optional files

| File | Purpose |
|------|---------|
| `genai_config.json` | Source for model metadata and search settings. If missing, a default config is generated from the merged graph. |
| `adapter.safetensors` | LoRA adapter weights; copied to the output when present. |
| `tokenizer/` | Not read by the converter; copy manually for inference. |
| `tokenizer/config.json` or `config.json` | Used to fill in `vocab_size` when the source genai config omits it. |

### What happens if files are missing?

The script validates the bundle **before** conversion:

- No decoder prefill/decode pair → error.
- Decoder pair present but embedding or lm head missing → error.
- Input path does not exist → error.

There is no partial conversion: all four components must be present.

## Conversion pipeline

The script runs five steps automatically:

1. Remove activation QuantizeLinear / DequantizeLinear nodes.
2. Promote activations to FP16.
3. Rewrite lm_head to emit pruned logits (`Gather` last token + `Unsqueeze` → shape `[1, 1, vocab]`).
4. Rebind prefill/decode seq_len axes to a dynamic `seq_len` parameter.
5. Fuse embedding + dynamic decoder + lm_head into one merged ONNX.

Pipeline **type** is inferred from the decoder prefill graph:

| Detected pattern | Behavior |
|------------------|----------|
| Quantized linear (MatMulNBits) | Standard QDQ removal and merge. |
| Many Gemm nodes, few MatMulNBits | Treats weights as folded Gemm (pure-Gemm / adapter path); may emit `lora_dequant.json`. |
| GroupQueryAttention with 8-bit KV cache | Preserves int8 KV path. |
| Many 2-bit MatMulNBits nodes | Low-bit decoder path. |

## Output files

| File | Description |
|------|-------------|
| `{decoder_stem}_merged.onnx` | Merged model (graph only or with inline weights). |
| `{decoder_stem}_merged.data` | External weight blob (when weights are externalized). |
| `genai_config.json` | GenAI runtime configuration (pipeline stages point at the merged file). |
| `lora_dequant.json` | Present only for folded-Gemm / adapter bundles. |
| `adapter.safetensors` | Copied from input when present and needed. |

The merged `genai_config.json` is adjusted for a single merged graph:

- Decoder inputs use `input_ids` (not precomputed embeddings).
- Execution provider profile is set to `hip` for AMDGPU stages.
- `vocab_size` is preserved from the source config or inferred from tokenizer metadata.

## Code layout

The five conversion steps map to modules under `merged_convert/`:

| Step | Module | Role |
|------|--------|------|
| 1–2 | `step1_qdq_fp16.py` | Remove activation Q/DQ; promote activations to fp16 (decoder / emb / lm_head) |
| 2 | `step2_fp16_cleanup.py` | FP16 activation cleanup helpers (also used as step 5 post-merge pass) |
| 3 | `step3_lm_head.py` | lm_head Gather + Unsqueeze → pruned logits `[1, 1, vocab]` |
| 4 | `step4_unfix_seq_len.py` | Rebind prefill/decode seq_len axes to a dynamic `seq_len` parameter (produces one dynamic decoder graph) |
| 5 | `step5_merge.py` | Fuse embedding + decoder + lm_head into one ONNX |

Supporting modules:

| Module | Role |
|--------|------|
| `qdq_ext.py` | Alternate Q/DQ + fp16 path for low-bit models |
| `int8kv.py` | Decoder conversion with int8 KV cache preserved |
| `bundle.py` | Input bundle detection and `genai_config.json` generation |
| `pipeline.py` | Wires the steps for each pipeline kind (`quantized_linear`, `int8_kv`, `low_bit`) |
| `cli.py` | Argument parsing and `main()` |

Run from this directory:

```bash
python run_merged_convert.py --input-dir ... --output-dir ...
```

Or as a module:

```bash
python -m merged_convert.cli --input-dir ... --output-dir ...
```

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `No decoder pair found` | Missing or misnamed `*_128.onnx` / `*_1.onnx` (or `*_0.onnx` / `*_1.onnx`) pair. |
| `Missing embedding or lm_head` | Emb or head file missing or name does not match glob patterns. |
| Runtime error on `vocab_size` | No vocab in source genai config and no `tokenizer/config.json`. Add one or fix paths. |
| GenAI rejects `embeddings` input | Re-run with a current script build; merged configs must use `input_ids`. |
