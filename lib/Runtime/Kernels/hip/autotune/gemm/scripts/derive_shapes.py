#!/usr/bin/env python3
"""Build shapes/gemm_shapes.csv from HuggingFace model configs + the legacy
OGA list + a set of boundary/stress shapes.

    python scripts/derive_shapes.py fetch    # HF config.json -> shapes/hf_model_configs.json
    python scripts/derive_shapes.py build    # that + oga_models.csv -> shapes/gemm_shapes.csv
    python scripts/derive_shapes.py budget   # print the sweep-size estimate

Mirrors ../matmul_nbits/scripts/extract_shapes.py in role: turn real model
geometry into the shape list the LUT sweep walks. `fetch` needs network access
(and HF_TOKEN for gated repos); `build` is offline and is what CI/regen runs.

Per transformer block, with hidden H, intermediate I, heads Q, kv-heads KV,
head_dim hd, vocab V, the GEMM nodes are:

  qkv_fused    N=(Q+2KV)*hd  K=H       q_proj    N=Q*hd   K=H
  kv_proj      N=2*KV*hd     K=H       attn_out  N=H      K=Q*hd
  mlp_gate_up  N=2I          K=H       mlp_up    N=I      K=H
  mlp_down     N=H           K=I       vocab     N=V      K=H

Both fused and split forms are emitted: exporters differ (ONNX/OGA fuses qkv
and gate_up, a plain HF export does not) and they are different shapes for tile
selection. MoE experts use moe_intermediate_size in place of I.

M ladder is per category, not uniform. An lm_head is not run at M=4096 by any
real runtime and a MoE expert never sees the full token count, so a uniform
ladder would spend most of the sweep on points that are never queried. The top
of the ladder is 16384 because above that the winning tile converges
(skill `offline-autotune-lut` pitfall 2), and nearest-neighbour covers larger M.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # autotune/gemm
SHAPES = ROOT / "shapes"
CONFIGS_JSON = SHAPES / "hf_model_configs.json"
OGA_CSV = SHAPES / "oga_models.csv"
OUT_CSV = SHAPES / "gemm_shapes.csv"

K_MAX = 65535            # GemmTunePoint.k is uint16
N_MAX = 4_000_000        # n is uint32

REPOS = [
    # dense LLM
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-72B",
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
    "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-small-8k-instruct",
    "microsoft/Phi-3-medium-4k-instruct", "microsoft/Phi-4-mini-instruct",
    "microsoft/phi-4",
    "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B", "01-ai/Yi-6B", "01-ai/Yi-34B",
    "internlm/internlm2_5-7b", "internlm/internlm2_5-20b",
    "THUDM/glm-4-9b", "openbmb/MiniCPM-2B-sft-bf16", "tiiuae/falcon-7b",
    "stabilityai/stablelm-2-1_6b", "allenai/OLMo-2-1124-7B",
    "ibm-granite/granite-3.1-8b-instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-Nemo-Base-2407",
    "NousResearch/Meta-Llama-3.1-8B", "NousResearch/Llama-3.2-1B",
    "unsloth/Llama-3.3-70B-Instruct",
    # MoE
    "mistralai/Mixtral-8x7B-v0.1", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen2-57B-A14B",
    "deepseek-ai/DeepSeek-V2-Lite", "openai/gpt-oss-20b", "openai/gpt-oss-120b",
    "allenai/OLMoE-1B-7B-0924",
    # vision / multimodal
    "openai/clip-vit-large-patch14", "openai/clip-vit-base-patch16",
    "google/siglip-so400m-patch14-384", "google/siglip-base-patch16-224",
    "google/siglip2-so400m-patch14-384", "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct", "OpenGVLab/InternViT-300M-448px",
    "OpenGVLab/InternViT-6B-448px-V1-5",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    # speech / embedding
    "openai/whisper-large-v3", "BAAI/bge-large-en-v1.5", "BAAI/bge-m3",
    "intfloat/multilingual-e5-large", "sentence-transformers/all-MiniLM-L6-v2",
]

# Shapes whose family is a shipping target -> priority 1 (full ladder, 3 dtypes).
TIER1_MODELS = {
    "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen3-0.6B",
    "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Phi-3-mini-4k-instruct",
    "Phi-4-mini-instruct", "phi-4", "Phi-3-medium-4k-instruct",
    "Meta-Llama-3.1-8B", "Llama-3.2-1B", "Mistral-7B-v0.1",
    "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Llama-8B",
    "SmolLM2-1.7B", "gpt-oss-20b", "Qwen3-30B-A3B",
    "siglip-so400m-patch14-384", "clip-vit-large-patch14",
    "Qwen2.5-VL-7B-Instruct",
}

CFG_KEYS = ("hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "model_type",
            "moe_intermediate_size", "shared_expert_intermediate_size")

# Boundary / stress shapes. These are not "more models" -- they are the places
# `hip-gemm-kernel-perf` documents the kernel as weakest, so the ladder must be
# dense there or nearest-neighbour lands on the wrong side of a winner flip:
# pow2-K L1 aliasing (K%1024==0), ragged K (K%16==0 && K%32!=0), non-pow2 mid N.
STRESS = [
    (2816, 4096), (3840, 8192), (1536, 6144), (2816, 8192), (4608, 4608),
    (6144, 6144), (4096, 6144), (5120, 6144), (2048, 6144), (1024, 6144),
    (4096, 3072), (2048, 3072), (8192, 3072), (4096, 16384), (8192, 16384),
    (1152, 4304), (4304, 1152), (2048, 4304), (1280, 3420), (3420, 1280),
    (4096, 4304), (2560, 4304), (1024, 2064), (2064, 1024), (4096, 2064),
    (7168, 2048), (2048, 7168), (7168, 7168), (2304, 5760), (5760, 2304),
    (4544, 4544), (4864, 896), (896, 4864), (11008, 2048), (2048, 11008),
    (10944, 2048), (2048, 10944), (1408, 2048), (2048, 1408),
    (12800, 4096), (4096, 12800), (20480, 7168), (7168, 20480),
    (27648, 5120), (5120, 27648), (29568, 8192), (8192, 29568),
    (25600, 5120), (5120, 25600), (18944, 3584), (3584, 18944),
    (9728, 2560), (2560, 9728), (3072, 1024), (1024, 3072),
]

DECODE = [1, 4, 8]                                        # M < 16 -> GEMV
PREFILL_FULL = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
PREFILL_MID = [16, 32, 64, 128, 256, 512, 1024, 2048]
PREFILL_SMALL = [16, 32, 64, 128, 256, 512]
# Real vision-token counts, (image/patch)^2: CLIP-B/16@224=196,
# CLIP-L/14@224=256, SigLIP-so400m/14@384=729, InternViT@448/14=1024;
# Qwen2.5-VL is dynamic, 256..4096 typical.
VISION = [196, 256, 576, 729, 1024, 2048, 4096]

LADDER = {
    # lm_head: runtimes project only the last token, at decode and at prefill.
    "vocab": DECODE + [16, 32, 128],
    # MoE expert: tokens_per_expert = M * topk / n_experts (Qwen3-30B-A3B: M/16).
    "moe": DECODE + PREFILL_SMALL,
    "vis": DECODE + VISION,
    "tiny": DECODE + PREFILL_SMALL,
    "default": DECODE + PREFILL_FULL,
}


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------

def cmd_fetch(args) -> int:
    ok, failed = {}, {}
    for repo in REPOS:
        url = f"https://huggingface.co/{repo}/raw/main/config.json"
        req = urllib.request.Request(url, headers={"User-Agent": "gemm-shapes/1"})
        if os.environ.get("HF_TOKEN"):
            req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                cfg = json.load(r)
        except Exception as e:  # noqa: BLE001
            failed[repo] = str(e)[:90]
            print(f"FAIL {repo}: {str(e)[:90]}", file=sys.stderr)
            continue
        out = {k: cfg[k] for k in CFG_KEYS
               if k in cfg and not isinstance(cfg[k], (dict, list))}
        for sub in ("text_config", "vision_config", "llm_config"):
            if isinstance(cfg.get(sub), dict):
                out.update({f"{sub}.{k}": cfg[sub][k] for k in CFG_KEYS
                            if k in cfg[sub]
                            and not isinstance(cfg[sub][k], (dict, list))})
        ok[repo] = out
        print(f"OK   {repo}")
    SHAPES.mkdir(parents=True, exist_ok=True)
    CONFIGS_JSON.write_text(
        json.dumps({"ok": ok, "failed": failed}, indent=1, sort_keys=True),
        encoding="utf-8")
    print(f"\n{len(ok)} ok, {len(failed)} failed -> {CONFIGS_JSON}")
    return 0


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def emit(rows, n, k, cat, src):
    if not n or not k or n <= 0 or k <= 0 or k > K_MAX or n > N_MAX:
        return
    rows[(int(n), int(k))].add((cat, src))


def from_block(rows, src, c, prefix="", cat_prefix=""):
    g = lambda key: c.get(prefix + key)  # noqa: E731
    H, I, Q = g("hidden_size"), g("intermediate_size"), g("num_attention_heads")
    if not H:
        return
    KV = g("num_key_value_heads") or Q
    hd = g("head_dim") or (H // Q if Q else None)
    V, p = g("vocab_size"), cat_prefix

    if Q and hd:
        emit(rows, (Q + 2 * KV) * hd, H, p + "qkv_fused", src)
        emit(rows, Q * hd, H, p + "q_proj", src)
        if KV != Q:
            emit(rows, 2 * KV * hd, H, p + "kv_proj", src)
        emit(rows, H, Q * hd, p + "attn_out", src)
    if I:
        emit(rows, 2 * I, H, p + "mlp_gate_up", src)
        emit(rows, I, H, p + "mlp_up", src)
        emit(rows, H, I, p + "mlp_down", src)
    if V and not cat_prefix:
        emit(rows, V, H, "vocab", src)
    for key, tag in (("moe_intermediate_size", "moe"),
                     ("shared_expert_intermediate_size", "moe_shared")):
        if (mi := g(key)):
            emit(rows, 2 * mi, H, f"{tag}_gate_up", src)
            emit(rows, mi, H, f"{tag}_up", src)
            emit(rows, H, mi, f"{tag}_down", src)
    if c.get("model_type") == "gpt_oss" and I:      # experts reuse I == H
        emit(rows, 2 * I, H, "moe_gate_up", src)
        emit(rows, H, I, "moe_down", src)


def ladder_for(category: str, priority: int) -> list[int]:
    c = category.split("|")[0]
    key = ("vocab" if c == "vocab" else
           "moe" if c.startswith("moe") else
           "vis" if c.startswith("vis") else
           "tiny" if c == "tiny" else "default")
    ms = LADDER[key]
    if priority == 2:                                  # broad coverage, cheaper
        ms = [m for m in ms if m in DECODE or m in PREFILL_MID]
    return ms


def cmd_build(args) -> int:
    rows: dict[tuple[int, int], set] = defaultdict(set)
    data = json.loads(CONFIGS_JSON.read_text(encoding="utf-8"))
    for repo, c in data["ok"].items():
        src = repo.split("/")[-1]
        from_block(rows, src, c)
        for sub, tag in (("text_config.", ""), ("vision_config.", "vis_"),
                         ("llm_config.", "")):
            if any(k.startswith(sub) for k in c):
                from_block(rows, src, c, prefix=sub, cat_prefix=tag)

    with open(OGA_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            emit(rows, int(r["N"]), int(r["K"]), r["category"], "oga")
    for n, k in STRESS:
        emit(rows, n, k, "stress", "boundary")

    out = []
    for (n, k), tags in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cats = sorted({c for c, _ in tags})
        srcs = sorted({s for _, s in tags})
        cat = cats[0] if len(cats) == 1 else "|".join(cats[:3])
        priority = 1 if ("oga" in srcs or TIER1_MODELS & set(srcs)) else \
               3 if "boundary" in srcs else 2
        out.append({"N": n, "K": k, "category": cat, "priority": priority,
                    "source": ";".join(srcs[:4]),
                    "m_list": " ".join(str(m) for m in ladder_for(cat, priority))})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["N", "K", "category", "priority",
                                          "source", "m_list"])
        w.writeheader()
        w.writerows(out)
    per_tier = defaultdict(int)
    for r in out:
        per_tier[r["priority"]] += 1
    print(f"{len(out)} shapes -> {OUT_CSV}")
    for t in sorted(per_tier):
        print(f"  priority{t}: {per_tier[t]}")
    return 0


# --------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------

def cmd_budget(args) -> int:
    rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8")))
    per = defaultdict(lambda: {"shapes": 0, "dec": 0, "pre": 0})
    for r in rows:
        t = int(r["priority"])
        ms = [int(m) for m in r["m_list"].split()]
        per[t]["shapes"] += 1
        # decode: GemvNt + GemvNn, x {f16, bf16, f32}
        per[t]["dec"] += len([m for m in ms if m < 16]) * 2 * 3
        # prefill: Wmma x {f16, bf16} NT  +  TiledFma f32 x {NT, NN}
        per[t]["pre"] += len([m for m in ms if m >= 16]) * 4

    print(f"{'priority':<6}{'shapes':>8}{'decode pts':>12}{'prefill pts':>13}")
    td = tp = 0
    for t in sorted(per):
        v = per[t]
        td, tp = td + v["dec"], tp + v["pre"]
        print(f"{t:<6}{v['shapes']:>8}{v['dec']:>12}{v['pre']:>13}")
    print(f"{'ALL':<6}{len(rows):>8}{td:>12}{tp:>13}")
    print(f"\ndecode  {td} pts x ~0.2 s = {td * 0.2 / 60:.0f} min")
    for lbl, s in (("optimistic 3 s", 3), ("likely 8 s", 8), ("worst 20 s", 20)):
        print(f"prefill {tp} pts, {lbl:<15} = {tp * s / 3600:5.1f} h")
    print(f"\ntier1 only, prefill @8 s = {per[1]['pre'] * 8 / 3600:.1f} h")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fetch", "build", "budget"])
    args = ap.parse_args()
    return {"fetch": cmd_fetch, "build": cmd_build,
            "budget": cmd_budget}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
