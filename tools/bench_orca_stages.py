#!/usr/bin/env python3
"""Attribute the clean (PERF=0) decode ms/token across the 3 sub-sessions and
compare each stage to its memory-bandwidth floor, to show whether decode is
bandwidth-bound or dispatch/overhead-bound.

Usage: bench_orca_stages.py <model_dir> <ep_dll>
Env:   BENCH_DECODE_STEPS (default 55), BENCH_WARMUP (default 5),
       MEAS_BW_GBS (measured read BW, default 247)
"""
import os, sys, time, statistics, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_orca_2bit import OrcaSplitSession

def weight_bytes(cfg):
    """Approx weight+scale+zp bytes read per token, per stage."""
    H = cfg["hidden_size"]; I = cfg["intermediate_size"]; L = cfg["num_hidden_layers"]
    KV = cfg["num_key_value_heads"] * cfg["head_dim"]; V = cfg["vocab_size"]
    bs = 32  # decode block_size (from PERF trace)
    def q(nw, bits):  # bytes for nw weights at `bits` + fp16 scales + 4-bit zp, block=bs
        return nw*bits/8 + (nw/bs)*2 + (nw/bs)*0.5
    per_layer = q(H*H,2) + 2*q(H*KV,2) + q(H*H,2) + 2*q(H*I,2) + q(I*H,2)  # q,k,v,o,gate,up,down
    body = per_layer * L
    lm   = q(H*V, 4)
    return body, lm

def main():
    model_dir = Path(sys.argv[1]); ep_dll = sys.argv[2]
    n = int(os.environ.get("BENCH_DECODE_STEPS","55")); warm = int(os.environ.get("BENCH_WARMUP","5"))
    BW = float(os.environ.get("MEAS_BW_GBS","247"))  # GB/s measured read

    sess = OrcaSplitSession(model_dir, ep_dll, max_seq_len=1024)
    cfg = json.load(open(model_dir/"config.json"))
    body_B, lm_B = weight_bytes(cfg)

    # prime
    ids = np.array([list(range(1,129))], dtype=np.int64)
    h = sess.prefill(sess.embed(ids), 0); past = 128
    tok = int(np.argmax(sess.lm_head(h[:,-1:,:])[0,-1]))

    te, tb, tl = [], [], []
    for _ in range(n):
        x = np.array([[tok]], dtype=np.int64)
        t0=time.perf_counter(); h=sess.embed(x);            t1=time.perf_counter()
        h=sess.decode_step(h, past);                        t2=time.perf_counter()
        lg=sess.lm_head(h);                                 t3=time.perf_counter()
        tok=int(np.argmax(lg[0,-1])); past+=1
        te.append((t1-t0)*1e3); tb.append((t2-t1)*1e3); tl.append((t3-t2)*1e3)

    def med(a): return statistics.median(a[warm:])
    e,b,l = med(te), med(tb), med(tl)
    tot = e+b+l
    print("\n=========== CLEAN PER-STAGE DECODE BREAKDOWN (median ms/token) ===========")
    print(f"{'stage':<14}{'ms':>8}{'% tok':>8}{'bytes':>10}{'floor ms':>10}{'eff %':>8}")
    for name, ms, B in [("embed",e,0.05e9),("decode-body",b,body_B),("lm_head",l,lm_B)]:
        floor = B/BW/1e6  # GB/(GB/s) -> ms ; B in bytes -> /1e9 s *1e3 ms = /1e6
        eff = 100*floor/ms if ms>0 else 0
        print(f"{name:<14}{ms:>8.2f}{100*ms/tot:>7.1f}%{B/1e9:>9.2f}G{floor:>10.2f}{eff:>7.1f}%")
    floor_tot = (0.05e9+body_B+lm_B)/BW/1e6
    print(f"{'TOTAL':<14}{tot:>8.2f}{100:>7.1f}%{(0.05e9+body_B+lm_B)/1e9:>9.2f}G{floor_tot:>10.2f}{100*floor_tot/tot:>7.1f}%")
    print(f"\nMeasured: {tot:.1f} ms/token = {1000/tot:.2f} tok/s")
    print(f"Roofline: {floor_tot:.1f} ms/token = {1000/floor_tot:.2f} tok/s  (@ {BW:.0f} GB/s read)")
    print(f"=> running at {100*floor_tot/tot:.0f}% of the memory roofline; "
          f"{tot-floor_tot:.0f} ms/token ({100*(tot-floor_tot)/tot:.0f}%) is NON-bandwidth overhead")
    print("=========================================================================")

if __name__ == "__main__":
    main()
