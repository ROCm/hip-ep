// ============================================================
// Does the decode winner depend on max_seq, or only on seq_kv?
//
// max_seq is the KV cache capacity: the cache is [B, G, max_seq, d], so head g
// starts at g * max_seq * d. The work length is read from seqlens_k, so the two
// are independent -- a given seq_kv touches the same number of bytes no matter
// how large the backing cache is. Only the spacing between heads changes.
//
// That makes max_seq a candidate for being dropped from the LUT key: keeping it
// multiplies the table by the number of cache capacities a deployment might use.
// This probe measures whether dropping it costs anything, by sweeping max_seq at
// a fixed seq_kv and asking two questions per (geometry, seq_kv):
//
//   1. does the fastest config change as max_seq grows?
//   2. if we pick the config that won at max_seq == seq_kv and use it at every
//      other max_seq, how much slower is it than that max_seq's own winner?
//
// Question 2 is the one that matters: a winner that flips between two configs
// that are within noise of each other costs nothing.
//
// Because this file owns the kernel TU, do NOT link gqa_kernel.obj with it.
// ============================================================

#include "../../../../hip/gqa_kernel.hip"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define PROBE_CHECK(expr)                                                      \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),         \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

struct Geometry {
  int H, G, d;
  const char *label;
};

struct Sample {
  int H, G, d, seq_kv, max_seq;
  int impl_wmma, splits;
  double ms;
};

// Median of `rounds` medians, each over `iters` back-to-back launches. Launching
// without an intervening sync amortises the launch cost the same way for every
// candidate, which is what keeps the comparison fair.
template <typename Launch>
double timeConfig(Launch &&launch, double target_ms, int rounds) {
  launch();
  PROBE_CHECK(hipDeviceSynchronize());

  hipEvent_t ev0, ev1;
  PROBE_CHECK(hipEventCreate(&ev0));
  PROBE_CHECK(hipEventCreate(&ev1));

  PROBE_CHECK(hipEventRecord(ev0));
  launch();
  PROBE_CHECK(hipEventRecord(ev1));
  PROBE_CHECK(hipEventSynchronize(ev1));
  float probe_ms = 0.0f;
  PROBE_CHECK(hipEventElapsedTime(&probe_ms, ev0, ev1));

  int iters = probe_ms > 1e-6f ? (int)(target_ms / probe_ms) : 200;
  iters = std::max(20, std::min(iters, 2000));

  std::vector<double> per_round;
  for (int r = 0; r < rounds; ++r) {
    launch();
    PROBE_CHECK(hipDeviceSynchronize());
    PROBE_CHECK(hipEventRecord(ev0));
    for (int i = 0; i < iters; ++i)
      launch();
    PROBE_CHECK(hipEventRecord(ev1));
    PROBE_CHECK(hipEventSynchronize(ev1));
    float ms = 0.0f;
    PROBE_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    per_round.push_back((double)ms / iters);
  }
  PROBE_CHECK(hipEventDestroy(ev0));
  PROBE_CHECK(hipEventDestroy(ev1));

  std::sort(per_round.begin(), per_round.end());
  return per_round[per_round.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  const char *out_path = nullptr;
  int rounds = 3;
  double target_ms = 30.0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--csv" && i + 1 < argc)
      out_path = argv[++i];
    else if (a == "--rounds" && i + 1 < argc)
      rounds = std::atoi(argv[++i]);
    else if (a == "--target-ms" && i + 1 < argc)
      target_ms = std::atof(argv[++i]);
  }

  // One geometry per decode kernel family: HpG=8/d=64 and HpG=4/d=128 have a
  // WMMA path, HpG=2/d=256 is scalar-only and has the largest cache footprint
  // (the case where address spacing is most likely to show up).
  const Geometry geoms[] = {
      {64, 8, 64, "gpt-oss 64:8:64"},
      {32, 8, 128, "llama 32:8:128"},
      {8, 4, 256, "gemma3 8:4:256"},
  };
  const int seq_kvs[] = {2048, 8192, 32768};
  const int kMaxCap = 131072;
  const int B = 1;
  const int cap = kFlashDecodeMaxSplits;

  std::vector<Sample> samples;

  for (const Geometry &g : geoms) {
    const int hpg = g.H / g.G;
    const float scale = 1.0f / std::sqrt((float)g.d);

    // Allocate once at the largest capacity, then vary the max_seq the kernel is
    // told about. Reallocating per max_seq would confound the measurement with
    // whatever physical pages the allocator happened to hand back.
    const size_t qn = (size_t)B * g.H * g.d;
    const size_t kn = (size_t)B * g.G * kMaxCap * g.d;
    const size_t pn = (size_t)B * g.H * cap * (g.d + 2);

    std::mt19937 rng(4321u + (unsigned)g.d);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> hQ(qn), hK(kn), hV(kn);
    for (auto &x : hQ)
      x = __float2half(dist(rng));
    for (auto &x : hK)
      x = __float2half(dist(rng));
    for (auto &x : hV)
      x = __float2half(dist(rng));

    __half *dQ, *dK, *dV, *dO;
    float *dPart;
    int *dSeq;
    PROBE_CHECK(hipMalloc(&dQ, qn * sizeof(__half)));
    PROBE_CHECK(hipMalloc(&dK, kn * sizeof(__half)));
    PROBE_CHECK(hipMalloc(&dV, kn * sizeof(__half)));
    PROBE_CHECK(hipMalloc(&dO, qn * sizeof(__half)));
    PROBE_CHECK(hipMalloc(&dPart, pn * sizeof(float)));
    PROBE_CHECK(hipMalloc(&dSeq, (size_t)B * sizeof(int)));
    PROBE_CHECK(
        hipMemcpy(dQ, hQ.data(), qn * sizeof(__half), hipMemcpyHostToDevice));
    PROBE_CHECK(
        hipMemcpy(dK, hK.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
    PROBE_CHECK(
        hipMemcpy(dV, hV.data(), kn * sizeof(__half), hipMemcpyHostToDevice));

    fprintf(stderr, "\n=== %s (HpG=%d) ===\n", g.label, hpg);

    for (int seq_kv : seq_kvs) {
      // The kernels read seqlens_k + 1, so this is what sets the work length.
      const int seqlen = seq_kv - 1;
      PROBE_CHECK(hipMemcpy(dSeq, &seqlen, sizeof(int), hipMemcpyHostToDevice));

      for (int max_seq = seq_kv; max_seq <= kMaxCap; max_seq *= 2) {
        int splits_set[7];
        int n_splits = 0;
        flashDecodeCandidateSplits(seq_kv, cap, splits_set, n_splits);
        const bool wmma_ok = flash_decode_wmma_supported(g.d, hpg);

        for (int impl = 0; impl < 2; ++impl) {
          const bool use_wmma = (impl == 1);
          if (use_wmma && !wmma_ok)
            continue;
          for (int si = 0; si < n_splits; ++si) {
            const FlashDecodeCfg cfg{use_wmma, splits_set[si]};
            const double ms = timeConfig(
                [&]() {
                  launchFlashDecodeConfig(cfg, nullptr, dQ, dK, dV, dO, dPart, B,
                                          g.H, g.G, g.d, hpg, max_seq, scale,
                                          dSeq, /*local_window_size=*/-1,
                                          /*hSink=*/nullptr,
                                          /*use_smooth_softmax=*/0);
                },
                target_ms, rounds);
            samples.push_back({g.H, g.G, g.d, seq_kv, max_seq,
                               use_wmma ? 1 : 0, splits_set[si], ms});
          }
        }

        // Report the winner for this (seq_kv, max_seq) as we go.
        const Sample *best = nullptr;
        for (const Sample &s : samples)
          if (s.H == g.H && s.G == g.G && s.d == g.d && s.seq_kv == seq_kv &&
              s.max_seq == max_seq && (!best || s.ms < best->ms))
            best = &s;
        fprintf(stderr, "  seq_kv=%6d max_seq=%7d -> %s_SPLITS%-3d %.5f ms\n",
                seq_kv, max_seq, best->impl_wmma ? "wmma" : "scalar",
                best->splits, best->ms);
      }
    }

    hipFree(dQ);
    hipFree(dK);
    hipFree(dV);
    hipFree(dO);
    hipFree(dPart);
    hipFree(dSeq);
  }

  FILE *out = out_path ? fopen(out_path, "w") : stdout;
  if (!out) {
    fprintf(stderr, "cannot open %s\n", out_path);
    return 1;
  }
  fprintf(out, "H,G,d,seq_kv,max_seq,impl,splits,config,ms\n");
  for (const Sample &s : samples)
    fprintf(out, "%d,%d,%d,%d,%d,%s,%d,%s_SPLITS%d,%.6f\n", s.H, s.G, s.d,
            s.seq_kv, s.max_seq, s.impl_wmma ? "wmma" : "scalar", s.splits,
            s.impl_wmma ? "wmma" : "scalar", s.splits, s.ms);
  if (out_path) {
    fclose(out);
    fprintf(stderr, "\nwrote %s (%zu rows)\n", out_path, samples.size());
  }
  return 0;
}
