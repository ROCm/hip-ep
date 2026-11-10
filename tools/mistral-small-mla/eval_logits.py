#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Teacher-forced next-token evaluation of an OGA model on natural text.

Prefills <ctx> tokens of a text file, then appends the reference text one
token at a time and records the logits that predict each next token. The
logits (fp16) and reference ids go to an .npz, so runs of two models (or two
builds) can be compared:

  eval_logits.py <model dir> <out.npz> --text FILE [--ctx 256] [--n 1024] [--bos]
  eval_logits.py --compare <base.npz> <other.npz> [...]

--compare prints NLL, perplexity, top-1 accuracy, top-1 agreement with the
first run and the mean / max KL(base || run) over the rows where every run's
logits are finite, plus each run's count of non-finite rows.
"""

import argparse
import sys

import numpy as np


def run(a):
    import onnxruntime_genai as og

    cfg = og.Config(a.model)
    cfg.clear_providers()
    cfg.append_provider("AMDGPU")
    cfg.set_provider_option("AMDGPU", "profile", "hip")
    model = og.Model(cfg)
    tok = og.Tokenizer(model)
    with open(a.text, encoding="utf-8") as f:
        ids = ([1] if a.bos else []) + list(tok.encode(f.read()))
    ids = ids[: a.ctx + a.n + 1]
    if len(ids) != a.ctx + a.n + 1:
        sys.exit(f"error: {a.text} has only {len(ids)} tokens, need {a.ctx + a.n + 1}")
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=a.max_length or a.ctx + a.n + 16, do_sample=False
    )
    gen = og.Generator(model, params)
    gen.append_tokens(ids[: a.ctx])
    rows = []
    for i in range(a.n):
        logits = np.asarray(gen.get_output("logits"), dtype=np.float32).reshape(-1)
        rows.append(logits[-a.vocab :].astype(np.float16))
        gen.append_tokens([ids[a.ctx + i]])
    np.savez(a.out, logits=np.stack(rows), target=np.array(ids[a.ctx : a.ctx + a.n]))
    print(f"saved {len(rows)} x {a.vocab} logits to {a.out}")


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def compare(paths):
    runs = [np.load(p) for p in paths]
    tgt = runs[0]["target"]
    ok = np.ones(len(tgt), bool)
    for d in runs:
        if not (d["target"] == tgt).all():
            sys.exit("error: the runs used different reference text")
        ok &= np.isfinite(d["logits"]).all(1)
    lb = logsoftmax(runs[0]["logits"][ok])
    pb = np.exp(lb)
    tgt = tgt[ok]
    rows = np.arange(len(tgt))
    print(f"{ok.sum()} of {len(ok)} rows finite in every run")
    print(
        f"{'run':40s} {'bad rows':>8s} {'NLL':>7s} {'PPL':>7s} {'top1 acc':>8s} "
        f"{'top1=base':>9s} {'KL(base||x)':>11s} {'max KL':>7s}"
    )
    for p, d in zip(paths, runs):
        bad = int((~np.isfinite(d["logits"]).all(1)).sum())
        lx = logsoftmax(d["logits"][ok])
        nll = -lx[rows, tgt].mean()
        acc = (lx.argmax(1) == tgt).mean()
        agree = (lx.argmax(1) == lb.argmax(1)).mean()
        kl = (pb * (lb - lx)).sum(1)
        print(
            f"{p[-40:]:40s} {bad:8d} {nll:7.4f} {np.exp(nll):7.3f} {acc:8.2%} {agree:9.2%} "
            f"{kl.mean():11.5f} {kl.max():7.3f}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare(sys.argv[2:])
        sys.exit(0)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--text", required=True, help="UTF-8 reference text")
    ap.add_argument("--ctx", type=int, default=256, help="prefilled context tokens")
    ap.add_argument("--n", type=int, default=1024, help="teacher-forced tokens")
    ap.add_argument("--vocab", type=int, default=131072, help="logits per row")
    ap.add_argument("--max-length", type=int, default=0)
    ap.add_argument(
        "--bos",
        action="store_true",
        help="prepend BOS (id 1), as the chat template does",
    )
    run(ap.parse_args())
