#!/usr/bin/env python3
"""ORCA 2-bit decode/prefill microbenchmark: best+median over N steps after warmup.

Reuses OrcaSplitSession from run_orca_2bit.py. Reports:
  - prefill ms/token (128-token prefill)
  - decode best / median / mean ms/tok and tok/s over BENCH_DECODE_STEPS,
    dropping the first BENCH_WARMUP steps (M=1 GEMV autotune warmup).

Usage: bench_orca.py <model_dir> <ep_dll>
Env:   BENCH_DECODE_STEPS (default 55), BENCH_WARMUP (default 5)
"""
import os, sys, time, statistics
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_orca_2bit import OrcaSplitSession

def main():
    model_dir = Path(sys.argv[1])
    ep_dll = sys.argv[2]
    n_decode = int(os.environ.get("BENCH_DECODE_STEPS", "55"))
    warmup   = int(os.environ.get("BENCH_WARMUP", "5"))
    prefill_len = 128

    print(f"[bench] loading sessions (ep_dll={ep_dll})", flush=True)
    sess = OrcaSplitSession(model_dir, ep_dll, max_seq_len=1024)

    # --- prefill (128 real tokens) ---
    ids = np.array([list(range(1, prefill_len + 1))], dtype=np.int64)
    t0 = time.perf_counter()
    h = sess.embed(ids)
    h = sess.prefill(h, 0)
    t1 = time.perf_counter()
    prefill_ms = (t1 - t0) * 1000.0
    past_len = prefill_len
    logits = sess.lm_head(h[:, -1:, :])
    next_token = int(np.argmax(logits[0, -1]))

    # --- decode loop (back-to-back, no sleeps) ---
    times = []
    for _ in range(n_decode):
        tok_ids = np.array([[next_token]], dtype=np.int64)
        t0 = time.perf_counter()
        h = sess.embed(tok_ids)
        h = sess.decode_step(h, past_len)
        logits = sess.lm_head(h)
        t1 = time.perf_counter()
        next_token = int(np.argmax(logits[0, -1]))
        past_len += 1
        times.append((t1 - t0) * 1000.0)

    kept = times[warmup:]
    best = min(kept); med = statistics.median(kept); mean = statistics.mean(kept)
    print("\n================ BENCH RESULTS ================")
    print(f"PREFILL: {prefill_ms:.1f} ms for {prefill_len} tokens "
          f"= {prefill_ms/prefill_len:.3f} ms/token")
    print(f"DECODE over {len(kept)} steps (dropped {warmup} warmup, {n_decode} total):")
    print(f"  best   : {best:6.2f} ms/tok  = {1000.0/best:6.2f} tok/s")
    print(f"  median : {med:6.2f} ms/tok  = {1000.0/med:6.2f} tok/s")
    print(f"  mean   : {mean:6.2f} ms/tok  = {1000.0/mean:6.2f} tok/s")
    print("===============================================")

if __name__ == "__main__":
    main()
