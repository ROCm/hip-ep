<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Mistral-Small-4 latent-cache (MLA) conversion

Offline rewrite of the pre-folded Mistral-Small-4-119B OGA export
(`Mistral-Small-4-119B-2603-fp16-ve-fp16-int4-rtn-gs128-hipep` with its QMoE
weights pre-folded) into a model whose KV cache holds the 320-wide MLA latent
row instead of 32 heads of expanded K/V. hip-ep fuses the resulting attention
into `hip.mla_attention`, which decodes over the latent cache directly.

On gfx1151 (Strix Halo), against the same model without the latent cache, this
gives:

- decode at a 16k text context: 28.6 instead of 70.8 ms per token (2.5x),
- decode at 2k: 26.6 instead of 31.0 ms per token,
- CI image benchmark with the 16k prompt (19.7k tokens): decode 29.1
  instead of 80.1 ms per token, peak GTT 62.5 instead of 78.7 GiB.

At 19.7k tokens, time to first token is 11% slower with the default
512-token prefill chunks; see [Prefill chunk size](#prefill-chunk-size).
At a 128-token text context, decode is 2% slower (26.1 vs 25.5 ms).

## What it does

`convert_mistral_mla.py` rewrites `decoder/model.onnx` only. It applies three
independent rewrites, each of which can be turned off:

1. **Residual scaling** (`--residual-scale`, default 8). For the BOS token,
   layer 32's MoE output (~1.5e5) overflows fp16; the inf turns into NaN
   logits for every prompt that starts with BOS, which is every chat prompt.
   RMSNorm is scale invariant, so the residual stream is carried divided by 8:
   `inputs_embeds` is multiplied by 1/8, the o_proj, shared-expert up_proj and
   QMoE fc1 up-row scales are divided by 8, and the skip-norm epsilons by 64.
   Every rescaled fp16 value stays exact. Measured on 1024 teacher-forced
   tokens without BOS, it matches the original within fp16 noise (perplexity
   6.865 vs 6.887, mean KL 0.0099).
2. **int8 lm_head** (`--lm-head-bits`, default 8). The fp16 lm_head MatMul
   becomes a symmetric round-to-nearest `MatMulNBits` with block 128. The
   relative weight RMS error is 0.68%.
3. **Static latent KV cache** (on unless `--no-mla`). Each of the 36
   `GroupQueryAttention` layers becomes:
   ```
   nonpad      = ReduceSum(attention_mask)
   present.key = TensorScatter(past.key, [c_norm (256) | k_pe (64)], nonpad - S)
   K, V        = up-project present.key with kv_b_proj's k_nope / v rows
   out         = Attention(q, K, V, nonpad_kv_seqlen = nonpad, is_causal = 1)
   ```
   The cache is a fixed `[1, 1, max_length, 320]` buffer shared between past
   and present. The converter sets `num_key_value_heads = 1`,
   `head_size = 320` and `past_present_share_buffer = true` in
   `genai_config.json`. The model is still plain ONNX (opset 24) and runs
   unfused on any EP that implements `TensorScatter` and `Attention` with
   `nonpad_kv_seqlen`. hip-ep's `hip-fusion-transform` replaces the expansion
   and the `Attention` with one `hip.mla_attention` per layer.

`--no-mla` applies only the first two rewrites. That gives the matching
baseline for A/B comparisons: same residual scaling and lm_head, standard
cache.

## Requirements

- Python 3.10 or newer with `pip install -r requirements.txt` (`onnx`,
  `numpy`).
- The **pre-folded** export as input: its QMoE fc1/fc2 weights and scales are
  initializers (offline-folded Reshape/Transpose chains) rather than computed
  in the graph. The tool refuses the unfolded export.
- The destination must be on the same filesystem as the source. The
  decoder's external data (~100 GB) is hard-linked, because ORT rejects
  external data that resolves outside the model directory. `--copy-data`
  copies it instead.
- Running the result on hip-ep needs a hip-ep build with
  `hip.mla_attention` (this directory's PR or later).

## Usage

```bash
pip install -r requirements.txt

# MLA model (residual scale 8, int8 lm_head, static latent cache)
python convert_mistral_mla.py <prefolded model dir> <mla model dir>

# Matching baseline (same residual scaling and lm_head, original attention)
python convert_mistral_mla.py <prefolded model dir> <baseline model dir> --no-mla

# Larger prefill chunks (see below)
python convert_mistral_mla.py <prefolded model dir> <dir> --chunk-size 2048
```

A conversion takes about 15 s and writes about 1.8 GB of new data. Expected
output:

```
residual scale 1/8: 108 scale tensors rescaled, 72 norm epsilons divided by 64
lm_head [4096x131072] int8 block 128: relative weight RMS error 0.6810%
latent cache: rewrote 36 attention layers, mla_latent.data 28 MiB
wrote <mla model dir> (1603 decoder nodes)
```

The destination contains:
- symlinks to every top-level entry of the source except `decoder/` and the
  `genai_config*.json` files;
- rewritten `genai_config*.json` files;
- `decoder/model.onnx`;
- hard links to the decoder's original data files;
- the new data files `decoder/res.data`, `decoder/lmhead_q8.data` and, for
  the MLA model, `decoder/mla_latent.data`.

The tool never writes into the source. It checks its input before creating
anything, and refuses an already converted or unfolded model.

## Checking that hip-ep fuses all layers

Dump the IR after each pass and count the fused ops in the decoder's
`hip-fusion-transform` output. All 36 layers should be `hip.mla_attention`,
with no `hip.kv_attention` left:

```bash
HIPDNN_EP_IR_DUMP_PATH=/tmp/irdump HIPDNN_EP_IR_DUMP_TREE=1 HIPDNN_EP_IR_DUMP_AFTER_ONLY=1 \
  python bench_decode.py -m <mla model dir> -p 128 -g 8
f=$(grep -l 'hip.mla_attention(' $(find /tmp/irdump* -name '*hip-fusion-transform*') | head -1)
grep -c 'hip.mla_attention(' "$f"   # 36
grep -c 'hip.kv_attention(' "$f"    # 0
```

## Measuring

`bench_decode.py` is a text-only prefill and decode benchmark at fixed past
lengths. Rep 0 of each length includes one-time compilation; use rep 1 or
later:

```bash
python bench_decode.py -m <model dir> -p 128 2048 16384 --max-length 16512 -g 64 -r 2
```

`eval_logits.py` records teacher-forced logits, so two models (or two builds)
can be compared by NLL, top-1 agreement and KL:

```bash
python eval_logits.py <baseline model dir> base.npz --text <long text file> --bos
python eval_logits.py <mla model dir> mla.npz --text <long text file> --bos
python eval_logits.py --compare base.npz mla.npz
```

## Prefill chunk size

OGA prefills in chunks of `search.chunk_size` tokens: 512 in the export. For
each chunk, `hip.mla_attention` expands all valid cache rows into per-head K/V
again. So the extra expansion work grows with the square of the prompt
length: about 2.2 s at 16k with 512-token chunks.

Larger chunks reduce it in proportion and also speed up the rest of prefill:
the MoE layers stream their expert weights once per chunk. Measured time to
first token, with `dog.jpg` and a 16k prompt (19.7k tokens):

| chunk_size | baseline | MLA |
|---|---|---|
| 512 | 29.1 s | 31.7 s |
| 2048 | 22.2 s | 22.7 s |
| 4096 | not measured | 20.5 s |

Peak GTT grows by less than 1 GiB from 512 to 4096.

## Limitations

- **Prefill re-expansion.** Prefill expands the cache once per chunk, as
  described above.
- **Batch.** Prefill needs one valid length for the whole batch.
- **Duplicate value cache.** genai allocates a value cache next to every key
  cache. The graph never reads it, but `TensorScatter` keeps it in sync,
  which costs 640 B per token and layer.
- **Numerics.** The fused decode reorders the math (absorbed `W_k` / `W_v`),
  so logits are not bit-identical to the baseline. With BOS, over 1024
  teacher-forced tokens of the CI's `prompt_16k.txt`, the MLA model's mean
  KL against the baseline is 0.012, top-1 agreement 95.7% and perplexity
  6.97 vs 6.88. For scale, only running the original model's `kv_b_proj`
  as an fp16 MatMul (same weights; measured without BOS) shifts its logits
  by a mean KL of 0.020: the
  top-4 MoE routing turns small rounding differences into a few high-KL
  tokens. Repeated runs of one model are bit-identical.
