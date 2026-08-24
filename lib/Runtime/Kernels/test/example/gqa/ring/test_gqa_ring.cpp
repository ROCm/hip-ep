// ============================================================
// custom_kernels KV-cache *ring append* test.
//
// Pins the wrap arithmetic of hip_gqa_kv_cache_append(..., ring=1), which backs
// a sliding-window layer whose cache is right-sized to the window instead of
// max_length. The buffer then holds only the newest present_seq positions and
// position p lives at slot p % present_seq.
//
// Two properties, and they fail in different ways:
//
//   * A prefill writes each slot EXACTLY ONCE, by writing only the newest
//     `capacity` positions. This is a correctness condition, not a tidiness
//     one: a windowed prefill hands the kernel far more positions than the
//     ring has cells, and writing them all would leave several source rows
//     racing for one slot with the last wave to retire deciding the winner.
//     The check is that each slot ends up holding the position the drop rule
//     assigns it -- under a race the surviving payload names some older
//     position instead.
//
//   * A decode lands at (total-1) % capacity. Off-by-one here is invisible
//     until the ring wraps, because before the first wrap slot and position
//     coincide.
//
// Each position carries its own index as its payload, so a slot holding the
// wrong position is caught directly rather than inferred from a numeric
// mismatch. The check is bit-exact: the append is pure relocation, so any
// difference at all means the wrong bytes moved.
//
// Build (from this directory):
//   hipcc -std=c++17 -O2 --offload-arch=gfx1151 \
//     test_gqa_ring.cpp ../../../../hip/gqa_kernel.hip \
//     -I../../../../include -o test_gqa_ring.exe
//   ./test_gqa_ring.exe
// ============================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" int hip_gqa_kv_cache_append(
    void* stream, const void* src, void* cache,
    int batch_size, int sq, int G, int d, int present_seq, int past_len,
    const void* seqlens_k, int element_size_bytes,
    int kv_dtype, const void* scale, int ring);

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),        \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

constexpr int kFp16 = 0; // hip_kv_dtype_t HIP_KV_DTYPE_FP16

int g_failures = 0;

// Sentinel for a slot the append never touched. Distinct from every payload,
// so "unwritten" and "wrong position" are different diagnoses.
constexpr float kUntouched = -1.0f;

// Payload for absolute position p. Every element of the row carries it, so a
// partially written row is caught too, not just a misplaced one.
float payload(int pos) { return static_cast<float>(pos + 1); }

// Recover the position a row claims to hold, or -2 if the row is not uniform
// (a torn write: two positions landed on one slot).
int claimed_position(const std::vector<__half>& cache, size_t row_base, int d) {
  const float first = __half2float(cache[row_base]);
  for (int i = 1; i < d; ++i)
    if (__half2float(cache[row_base + i]) != first)
      return -2;
  if (first == kUntouched)
    return -1;
  return static_cast<int>(first) - 1;
}

void report(const char* name, bool ok, const char* detail) {
  if (ok) {
    printf("  [ok]   %s\n", name);
    return;
  }
  printf("  [FAIL] %s: %s\n", name, detail);
  ++g_failures;
}

// Run one ring append and check every slot against the positions the ring is
// supposed to hold afterwards.
//
// past_len   absolute positions already in the cache
// sq         new tokens in this call
// capacity   ring cells (== the window)
//
// After the call the ring must hold the newest `capacity` positions of
// [0, past_len+sq), each at position % capacity. Positions older than that are
// dropped, and whatever the earlier state left in their slots stays -- the
// append promises nothing about a slot it is not the newest writer of, so the
// expectation below is built from that same rule rather than from "everything
// is fresh".
void check_ring_append(const char* name, int past_len, int sq, int capacity) {
  const int B = 1, G = 2, d = 8;
  const int total = past_len + sq;

  std::vector<__half> cache(static_cast<size_t>(B) * G * capacity * d,
                            __float2half(kUntouched));
  std::vector<__half> src(static_cast<size_t>(B) * sq * G * d);
  for (int s = 0; s < sq; ++s)
    for (int g = 0; g < G; ++g)
      for (int i = 0; i < d; ++i)
        src[(static_cast<size_t>(s) * G + g) * d + i] =
            __float2half(payload(past_len + s));

  void *d_cache = nullptr, *d_src = nullptr;
  HIP_CHECK(hipMalloc(&d_cache, cache.size() * sizeof(__half)));
  HIP_CHECK(hipMalloc(&d_src, src.size() * sizeof(__half)));
  HIP_CHECK(hipMemcpy(d_cache, cache.data(), cache.size() * sizeof(__half),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_src, src.data(), src.size() * sizeof(__half),
                      hipMemcpyHostToDevice));

  const int rc = hip_gqa_kv_cache_append(
      /*stream=*/nullptr, d_src, d_cache, B, sq, G, d, /*present_seq=*/capacity,
      past_len, /*seqlens_k=*/nullptr, /*element_size_bytes=*/2, kFp16,
      /*scale=*/nullptr, /*ring=*/1);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(cache.data(), d_cache, cache.size() * sizeof(__half),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_cache));
  HIP_CHECK(hipFree(d_src));

  char detail[256];
  if (rc != 0) {
    snprintf(detail, sizeof(detail), "append returned %d", rc);
    report(name, false, detail);
    return;
  }

  // The positions this append was responsible for writing, and where each
  // belongs. Anything older than total-capacity it must have skipped, so over
  // the surviving range p -> p % capacity is one-to-one and every entry below
  // is set by exactly one position.
  const int oldest_written = (total > capacity) ? total - capacity : 0;
  std::vector<int> expected(capacity, -1); // -1 = untouched by this append
  for (int p = past_len; p < total; ++p) {
    if (p < oldest_written)
      continue; // dropped on purpose: a newer position owns this slot
    expected[p % capacity] = p;
  }

  // Comparing against that expectation is also what detects the race the drop
  // rule exists to prevent. A kernel that wrote every position it was handed
  // would put several of them on one slot, and the surviving payload names
  // whichever wave retired last -- a position other than the newest, which is
  // a mismatch here. Dropping the rule from kv_ring_slot is caught on the
  // 2.5x-capacity prefill below as "slot 0 holds position 16, expected 32".
  for (int g = 0; g < G; ++g) {
    for (int slot = 0; slot < capacity; ++slot) {
      const size_t row = (static_cast<size_t>(g) * capacity + slot) * d;
      const int got = claimed_position(cache, row, d);
      if (got == -2) {
        snprintf(detail, sizeof(detail),
                 "head %d slot %d holds a torn row: two positions landed on "
                 "one slot",
                 g, slot);
        report(name, false, detail);
        return;
      }
      if (got != expected[slot]) {
        snprintf(detail, sizeof(detail),
                 "head %d slot %d holds position %d, expected %d "
                 "(past_len=%d sq=%d capacity=%d)",
                 g, slot, got, expected[slot], past_len, sq, capacity);
        report(name, false, detail);
        return;
      }
    }
  }
  report(name, true, "");
}

} // namespace

int main() {
  printf("ring append: slot = position %% capacity\n");

  // Decode, the common case: one token at (total-1) % capacity.
  // Before the first wrap slot and position coincide, so the interesting
  // instances are the ones at and after it.
  check_ring_append("decode below capacity  (past=5   sq=1 cap=16)", 5, 1, 16);
  check_ring_append("decode at the wrap     (past=16  sq=1 cap=16)", 16, 1, 16);
  check_ring_append("decode past the wrap   (past=37  sq=1 cap=16)", 37, 1, 16);
  check_ring_append("decode one before wrap (past=15  sq=1 cap=16)", 15, 1, 16);

  // Prefill, where the drop rule earns its keep: far more positions than cells.
  // Each writes every slot exactly once, which is the race property.
  check_ring_append("prefill 2.5x capacity  (past=0   sq=40 cap=16)", 0, 40, 16);
  check_ring_append("prefill exact capacity (past=0   sq=16 cap=16)", 0, 16, 16);
  check_ring_append("prefill under capacity (past=0   sq=9  cap=16)", 0, 9, 16);
  check_ring_append("prefill onto a wrapped ring (past=37 sq=20 cap=16)", 37, 20,
                    16);

  // A capacity that is not a power of two: `%` is exact, but an implementation
  // that reached for a mask instead would only pass above.
  check_ring_append("decode, odd capacity   (past=100 sq=1 cap=13)", 100, 1, 13);
  check_ring_append("prefill, odd capacity  (past=0  sq=31 cap=13)", 0, 31, 13);

  if (g_failures == 0) {
    printf("all ring append checks passed\n");
    return 0;
  }
  printf("%d ring append check(s) FAILED\n", g_failures);
  return 1;
}
