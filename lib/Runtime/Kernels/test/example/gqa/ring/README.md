# GQA ring-append test

Pins the wrap arithmetic of `hip_gqa_kv_cache_append(..., ring=1)`, the append
behind a sliding-window layer whose KV cache is right-sized to the window
instead of `max_length`. The buffer holds only the newest `present_seq`
positions, so position `p` lives at slot `p % present_seq`.

This is the cheapest place to pin that arithmetic: no ORT, no session, no
model. The numeric suite cannot reach it, because the ring requires `past` and
`present` to be the same allocation (`lib/Runtime/real/gqa.cpp`, the
`past_key != present_key` rejection) and that harness allocates every output
itself.

## Build and run

From this directory:

```bash
hipcc -std=c++17 -O2 --offload-arch=gfx1151 \
  test_gqa_ring.cpp ../../../../hip/gqa_kernel.hip \
  -I../../../../include -o test_gqa_ring.exe
./test_gqa_ring.exe
```

Substitute your own `--offload-arch`. Exit status is 0 when every check
passes, 1 otherwise, and each failure prints the head, the slot, the position
found and the position expected.

## What it checks

Each absolute position carries its own index as its payload, so a slot holding
the wrong position is reported directly rather than inferred from a numeric
mismatch. Comparison is bit-exact -- the append is pure relocation, so any
difference means the wrong bytes moved.

| Property | Why it can break |
|---|---|
| A decode lands at `(total-1) % capacity` | Off-by-one is invisible until the ring wraps, because before the first wrap slot and position coincide. |
| A prefill writes each slot exactly once | A windowed prefill is handed far more positions than the ring has cells. Writing them all races several source rows onto one slot, and the last wave to retire wins. |
| Positions older than `total - capacity` are dropped | That drop is what makes the mapping one-to-one, and hence what makes the prefill race-free. |
| `capacity` need not be a power of two | An implementation reaching for a bitmask instead of `%` passes every power-of-two case and fails these. |

The cases run at and around each boundary: one before the first wrap, exactly
at it, well past it, a prefill onto an already-wrapped ring, and the same at a
non-power-of-two capacity.

Removing the drop rule from `kv_ring_slot` -- the one line that makes the
prefill race-free -- fails the 2.5x-capacity prefill with `slot 0 holds
position 16, expected 32`, which is the negative control this test was checked
against.
