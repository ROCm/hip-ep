#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Text-only prefill and decode benchmark of an OGA model on hip-ep.

For each past length P: prefill a P-token prompt (repeated filler text), then
time G decode steps (generate_next_token) after W untimed warm-up steps and
report prefill time and steady-state decode time per token. Rep 0 of each
length includes one-time costs (compilation of new shapes, allocations); rep 1
and later are the warm numbers.

  bench_decode.py -m <model dir> [-p 128 2048 16384] [-g 64] [-r 2]
      [--max-length 16512]

The model runs on the AMD GPU umbrella EP with profile=hip (hip-ep); point
AMDGPU_EP_PATH at the libamdgpu-ep.so to test.
"""

import argparse
import statistics
import time

import onnxruntime_genai as og


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-p", "--past", type=int, nargs="+", default=[128, 2048, 16384])
    ap.add_argument("-g", "--gen", type=int, default=64, help="timed decode steps")
    ap.add_argument(
        "-w",
        "--warmup-decode",
        type=int,
        default=8,
        help="decode steps excluded from timing",
    )
    ap.add_argument("-r", "--reps", type=int, default=1)
    ap.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="KV cache length; default max(past) + gen + warmup + 16",
    )
    ap.add_argument(
        "--prompt",
        default="The quick brown fox jumps over the lazy dog. ",
        help="filler text, repeated to reach each past length",
    )
    ap.add_argument("--print-tokens", action="store_true")
    a = ap.parse_args()

    cfg = og.Config(a.model)
    cfg.clear_providers()
    cfg.append_provider("AMDGPU")
    cfg.set_provider_option("AMDGPU", "profile", "hip")
    t0 = time.perf_counter()
    model = og.Model(cfg)
    print(f"model load {time.perf_counter() - t0:.1f}s", flush=True)
    tok = og.Tokenizer(model)
    base_ids = list(tok.encode(a.prompt * 4096))

    max_len = a.max_length or (max(a.past) + a.gen + a.warmup_decode + 16)
    for past in a.past:
        for rep in range(a.reps):
            ids = base_ids[:past]
            assert len(ids) == past, (len(ids), past)
            params = og.GeneratorParams(model)
            params.set_search_options(
                max_length=max_len, min_length=max_len, do_sample=False
            )
            gen = og.Generator(model, params)
            t0 = time.perf_counter()
            gen.append_tokens(ids)
            t_prefill = time.perf_counter() - t0
            steps, toks = [], []
            for _ in range(a.warmup_decode + a.gen):
                t1 = time.perf_counter()
                gen.generate_next_token()
                toks.append(int(gen.get_next_tokens()[0]))
                steps.append(time.perf_counter() - t1)
            if a.print_tokens:
                print(f"text past={past}: {tok.decode(toks[:32])!r}")
            st = steps[a.warmup_decode :]
            med = statistics.median(st)
            print(
                f"past={past:6d} rep={rep} prefill {t_prefill * 1e3:8.1f} ms "
                f"({past / t_prefill:7.1f} tok/s) | decode median {med * 1e3:7.3f} ms "
                f"-> {1 / med:6.2f} tok/s, mean {statistics.mean(st) * 1e3:7.3f} ms, "
                f"min {min(st) * 1e3:6.2f} max {max(st) * 1e3:6.2f}",
                flush=True,
            )
            del gen


if __name__ == "__main__":
    main()
