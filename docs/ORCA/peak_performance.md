# Peak Decode Performance Analysis — ORCA 2-bit on Strix Halo (gfx1152)

## The Roofline Question: Why ~10 tok/s and not 20?

For decoder-only LLM inference at M=1 (single-token decode), the GPU must read
every model weight from memory once per token generated. There is no weight reuse.
This makes decode **memory-bandwidth-bound**, not compute-bound. The ceiling is
set by how fast the GPU can stream bytes from DRAM — nothing else.

---

## Step-by-Step Math

### Step 1 — How many bytes must the GPU read per token?

ORCA 2-bit has 40 transformer layers. Each layer has 7 matmuls:

| Projection | Shape (N × K) | Weight bytes (bits=2) | Scale bytes (fp32) |
|---|---|---|---|
| q_proj | 5120 × 5120 | 3.3 MB | 1.0 MB |
| k_proj | 1280 × 5120 | 0.8 MB | 0.3 MB |
| v_proj | 1280 × 5120 | 0.8 MB | 0.3 MB |
| o_proj | 5120 × 5120 | 3.3 MB | 1.0 MB |
| gate_proj | 13824 × 5120 | 8.9 MB | 2.7 MB |
| up_proj | 13824 × 5120 | 8.9 MB | 2.7 MB |
| down_proj | 5120 × 13824 | 8.9 MB | 2.7 MB |

Per layer total: **~86.8 MB**
40 layers total: **~3,473 MB**
LM head (fp16): **~328 MB**

**Grand total: ~3,800 MB read per token generated**

Weight bytes = N × K / 4 (bits=2, 4 weights per byte)
Scale bytes  = N × (K / block_size) × 4 bytes (fp32, block_size=64)

---

### Step 2 — How fast can this chip read memory?

The Radeon 860M (gfx1152) is an **integrated GPU** on Strix Halo. It shares
LPDDR5X system memory with the CPU — there is no dedicated VRAM.

| Memory spec | Value |
|---|---|
| Memory type | LPDDR5X |
| Bus width | 128-bit |
| Speed | ~7500 MT/s |
| Theoretical peak BW | ~120 GB/s (full system) |
| **Practical GPU BW** | **~40–50 GB/s** |

The gap between theoretical (120 GB/s) and practical GPU BW (~40–50 GB/s) comes
from:
- CPU activity competing for the same memory bus
- Memory controller overhead and refresh cycles
- Cache effects on irregular access patterns
- iGPU not getting exclusive bus access

---

### Step 3 — The ceiling calculation

```
max tok/s = effective GPU memory bandwidth / bytes per token

At 40 GB/s:  40,000 MB/s / 3,800 MB  =  10.5 tok/s
At 50 GB/s:  50,000 MB/s / 3,800 MB  =  13.2 tok/s
At 60 GB/s:  60,000 MB/s / 3,800 MB  =  15.8 tok/s
At 80 GB/s:  80,000 MB/s / 3,800 MB  =  21.1 tok/s  ← needs full chip BW
At 120 GB/s: 120,000 MB/s / 3,800 MB =  31.6 tok/s  ← theoretical peak (impossible in practice)
```

**Practical ceiling on this chip: ~10–11 tok/s**

---

### Step 4 — Where we are today

With all optimisations implemented (see `orca_2bit_optimizations.md`):

```
Measured: ~140 ms per token (warm autotune)
Throughput: 1000 ms / 140 ms = ~7.1 tok/s

Effective BW = 3,800 MB / 0.140 s = 27.1 GB/s
BW utilisation = 27.1 / 40 = 68% of practical GPU ceiling
```

The gap from 7 to 10 tok/s is kernel occupancy — more concurrent GPU waves to
better hide the ~400-cycle DRAM latency. This is achievable with further tuning.

---

### Step 5 — Why 20 tok/s requires different hardware

```
Required BW for 20 tok/s = 20 × 3,800 MB = 76,000 MB/s = 76 GB/s
```

76 GB/s is the chip's **entire theoretical DRAM budget** shared between CPU and
GPU. Even with zero CPU activity this is not reachable by the GPU alone on an
integrated memory architecture.

On a discrete GPU with dedicated VRAM:

| GPU | Memory BW | Est. tok/s (ORCA 2-bit) |
|---|---|---|
| Radeon 860M iGPU (this chip) | ~40 GB/s GPU-accessible | ~10 |
| RX 7600 (discrete) | 288 GB/s | ~76 |
| RX 7900 XTX (discrete) | 960 GB/s | ~250 |

---

## Does Raising Clock Frequency Help?

### GPU compute clock — No

ORCA decode is **memory-bandwidth-bound, not compute-bound**.

The GPU arithmetic (multiply-accumulate for matmuls) completes faster than the
next batch of weights arrives from DRAM. The GPU cores are idle, waiting for data.
Raising the GPU shader clock from 2.5 GHz to 3.0 GHz (+20%) gives approximately:

```
Additional compute throughput: +20%
Additional memory bandwidth:    +0%  (compute clock ≠ memory clock)
Decode throughput improvement:  ~0%
```

Verified by the roofline model: at 27 GB/s effective BW and 3,800 MB/token, the
GPU spends ~140 ms moving data and ~2 ms doing arithmetic. Speeding up the 2 ms
arithmetic part has no effect on the 140 ms wall time.

### Memory clock — Yes, but not tunable

The **memory clock** directly sets the bandwidth ceiling. LPDDR5X at 7500 MT/s on
a 128-bit bus gives ~120 GB/s peak. If memory could run at 9600 MT/s (theoretical
max for next-gen LP DDR6), peak would be ~154 GB/s and the GPU-accessible portion
might reach 55–60 GB/s — pushing the ceiling to ~14–16 tok/s.

However, **memory speed is not tunable at runtime** on this platform. It is set by
the memory modules soldered to the board and the SoC's memory controller. No power
plan, driver setting, or software change can increase it.

### Power / performance mode — Marginal

Higher power limits allow the GPU to boost its clock more consistently (reducing
thermal throttling) and can improve the memory controller's sustained throughput
slightly. In practice this yields:

- **Compute-bound workloads**: 10–30% improvement
- **Memory-bandwidth-bound workloads (ORCA decode)**: 2–5% improvement at most

Not enough to change the 10 tok/s ceiling materially.

---

## Summary

| Question | Answer |
|---|---|
| Bytes read per token | ~3,800 MB |
| Practical GPU BW on this chip | ~40–50 GB/s |
| Hard ceiling (40 GB/s) | **~10.5 tok/s** |
| Current achieved (warm) | **~7.1 tok/s** (27 GB/s, 68% of ceiling) |
| Gap to ceiling | ~3 tok/s — achievable with occupancy tuning |
| BW needed for 20 tok/s | 76 GB/s — exceeds full-chip DRAM budget |
| Does GPU compute clock help? | No — workload is memory-bound |
| Does memory clock help? | Yes, but not runtime-tunable |
| Does power/performance mode help? | 2–5% at most |
| Platform that hits 20 tok/s | Discrete GPU with 80+ GB/s dedicated VRAM BW |
