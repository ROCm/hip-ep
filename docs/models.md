<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Models

[`models.json`](models.json) lists the models for which AMD publishes a GPU EP export on Hugging Face, with the files needed to run each export. Every entry has the same fields, so that tools can read the file as well as people.

An export is listed once its Hugging Face repo is public, that is, once anyone can download it without logging in. The entries follow the order of the hip-ep model matrix, `site/_data/models.yml`: text models first, then vision-language models, each grouped by family. Within a family, older versions come first; within a version, smaller models come first, and a base model comes before a specialization of it.

## Example

The entry for Qwen3.6 35B-A3B:

```json
{
  "id": "qwen3.6-35b-a3b",
  "name": "Qwen3.6 35B-A3B",
  "family": "qwen",
  "model_type": "vlm",
  "base_model": "Qwen/Qwen3.6-35B-A3B",
  "params": "35B total, 3B active",
  "params_billions": {
    "total": 35,
    "active": 3
  },
  "architecture": "Gated DeltaNet + sparse mixture of experts + vision encoder",
  "task": "Image understanding, text generation",
  "license": "apache-2.0",
  "quant_scheme": "int4 RTN",
  "precision": "int4 RTN text decoder, group 32, with QMoE experts; fp16 vision encoder and embedding",
  "quantized_model": {
    "path": "https://huggingface.co/amd/Qwen3.6-35B-A3B_gpuep_1.0_gpu_128/tree/main",
    "hf_repo": "amd/Qwen3.6-35B-A3B_gpuep_1.0_gpu_128",
    "subdir": "",
    "onnx_files": {
      "text": "text.onnx",
      "vision": "vision.onnx",
      "embedding": "embedding.onnx"
    },
    "access": "public"
  }
}
```

## Fields

The file is a JSON object with four fields:

- `copyright` and `license`: the license header that other hip-ep files carry as a comment, given as fields because JSON has no comments. The MIT License covers this file only. Each model is distributed under its own license, which the `license` field of its entry names.
- `description`: a one-line summary of the file.
- `models`: the entries, one for each model.

Each entry has these fields:

| Field | Meaning | Examples |
|---|---|---|
| `id` | Lowercase slug of the model, unique in the file. For a model in `site/_data/models.yml`, the `id` used there. | `gpt-oss-120b`, `qwen3.6-35b-a3b` |
| `name` | Name of the base model, not of the export. | `GPT-OSS 120B`, `Qwen3.6 35B-A3B` |
| `family` | Model family. | `gpt-oss`, `qwen`, `gemma` |
| `model_type` | `llm` for a text model; `vlm` for a vision-language model, which also takes images as input. | `llm`, `vlm` |
| `base_model` | Hugging Face repo of the base model from which the export was made. | `Qwen/Qwen3.6-35B-A3B` |
| `params` | Parameter count of the base model, as its model card gives it. For a mixture of experts, the total count and the count that is active for each token. | `14.7B`, `35B total, 3B active` |
| `params_billions` | The same counts as numbers, in billions. | `{"total": 14.7}`, `{"total": 35, "active": 3}` |
| `architecture` | Structure of the base model. See [Architecture](#architecture). | `Dense transformer`, `Gated DeltaNet + sparse mixture of experts + vision encoder` |
| `task` | What the model is used for. | `Text generation`, `Code generation`, `Image understanding, text generation` |
| `license` | License of the base model and of the export. See [License](#license). | `apache-2.0`, `gemma` |
| `quant_scheme` | Weight type and quantization method of the text decoder of the export. See [Quantization](#quantization). | `int4 RTN`, `int4 k-quant`, `uint4 AWQ`, `int4` |
| `precision` | Precision of each part of the export. See [Quantization](#quantization). | `int4 RTN, group 128, asymmetric`, `int4 RTN text decoder, group 32, with QMoE experts; fp16 vision encoder and embedding` |
| `quantized_model` | The export. See [`quantized_model`](#quantized_model). | |

### Architecture

`architecture` is built from these terms:

- `Dense transformer`: each layer has a single feed-forward network.
- `Sparse mixture of experts`: the feed-forward network of each layer is a set of experts, of which a router selects a few for each token.
- `Gated DeltaNet +`: three of every four layers use Gated DeltaNet, a form of linear attention, instead of full attention.
- `+ vision encoder`: the model has a vision encoder, which turns images into input for the language model.

### License

`license` is the license identifier on the Hugging Face model card of the base model. The model card of the export gives the same identifier.

| Value | License |
|---|---|
| `apache-2.0` | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `gemma` | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |

Review the license of a model before you download or use it.

### Quantization

`quant_scheme` and `precision` describe the weights of the export. In every export, the activations are fp16.

- `int4` and `uint4` denote 4-bit weights, signed and unsigned, as the model card of the export names them.
- The quantization method follows the weight type when the model card names it. `RTN` (round to nearest) rounds each weight to the nearest 4-bit value. `AWQ` (activation-aware weight quantization) and `k-quant` search for the scales that reduce the quantization error. `int4` alone means that the model card does not name the method.
- `precision` describes each part of the export, separated by semicolons: the text decoder first, then the other parts. The words `text decoder` are omitted when the export has no other part.
- `group N` means that each group of N consecutive weights shares one scale. `asymmetric` means that each group also has a zero point; otherwise, the quantization is symmetric.
- `with QMoE experts` means that the expert weights of the mixture of experts are quantized in the same way and run as the `com.microsoft::QMoE` operator. `with symmetric QMoE experts` means the same, except that the expert weights have no zero points.
- `fp16` marks the parts whose weights are not quantized.

### `quantized_model`

`quantized_model` describes the AMD GPU EP export of the model. It has these fields:

| Field | Meaning |
|---|---|
| `path` | Hugging Face page of the folder to run: `https://huggingface.co/<hf_repo>/tree/main`, followed by `/<subdir>`, URL-encoded, when `subdir` is not empty. |
| `hf_repo` | Hugging Face repo of the export. |
| `subdir` | Folder of the repo that ONNX Runtime GenAI runs. It holds `genai_config.json`, the tokenizer and the ONNX files. `""` is the top of the repo. |
| `onnx_files` | The ONNX models that `genai_config.json` loads, by role, as paths relative to `subdir`. |
| `access` | `public`: anyone can download the export without logging in to Hugging Face. |

The roles in `onnx_files` are:

| Role | Model | Section of `genai_config.json` |
|---|---|---|
| `text` | Text decoder, which generates the output tokens. Every entry has one. | `decoder` |
| `vision` | Vision encoder, which turns an image into image features. Every `vlm` entry has one; no `llm` entry does. | `vision` |
| `embedding` | Embedding model, which merges the image features into the embeddings of the input tokens. Every `vlm` entry has one; no `llm` entry does. | `embedding` |

## Running an export

The `genai_config.json` of every export runs each of its models on the AMD GPU umbrella EP with `"provider_options": [{"AMDGPU": {"profile": "hip"}}]`, where the `hip` profile selects hip-ep, so an export runs without changes. Download an export with the Hugging Face CLI, for example:

```bash
hf download amd/Qwen3.6-35B-A3B_gpuep_1.0_gpu_128 --local-dir models/qwen3.6-35b-a3b
```

The `hf` command requires `huggingface_hub` 0.34 or later; with an earlier version, run `huggingface-cli download` with the same arguments. When `subdir` is not empty, add `--include "<subdir>/*"` and use `models/<id>/<subdir>` in place of `models/<id>`.

Then pass the folder to an ONNX Runtime GenAI application, for example `model_benchmark -i models/qwen3.6-35b-a3b`, set up as described in [OGA end-to-end benchmarking](quick_start.md#oga-end-to-end-benchmarking-with-model_benchmark).

## Adding, changing or removing a model

1. Add an entry when AMD makes the GPU EP export of a model public on Hugging Face, at the position that the order above gives it. Remove the entry when the export is no longer public.
2. Take the fields of the base model from its Hugging Face model card and `config.json`. For a model in `site/_data/models.yml`, use the `id`, `name`, `family` and `task` given there. Write `params` as the model card gives it, with `params_billions` to match; `architecture` from the layer types and the experts in `config.json`; `model_type` as `vlm` when `config.json` defines a vision encoder, otherwise `llm`; and `license` from the license identifier on the model card.
3. Take the fields of the export from its repo: the weight type and the method from the quantization section of its model card; the group sizes, the zero points and the QMoE experts from the `MatMulNBits` and `QMoE` nodes of its ONNX models; `subdir` from the folder that holds `genai_config.json`; `onnx_files` from the `filename` of each model in that file; and `path` from `hf_repo` and `subdir`. The model card of the export must give the same license as that of the base model. A model that `genai_config.json` loads from another section takes the name of that section as its role; add the role to the table above.
