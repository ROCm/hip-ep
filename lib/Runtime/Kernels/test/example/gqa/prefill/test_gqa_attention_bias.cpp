// ============================================================
// custom_kernels GQA attention-bias addressing test.
//
// Verifies hip_gqa_add_attention_bias_f32, the Step 8b kernel that adds an
// external additive mask onto the fp32 score matrix in the decomposed prefill
// pipeline (gqa.cpp).
//
// The score matrix is [total_heads, sq, total_seq] with a per-head stride of
// score_batch_stride, while the bias is [bias_batch, bias_heads, bias_sq,
// total_seq] and bias_batch / bias_heads may be 1 for ONNX-style broadcast.
// When the prefill is tiled over query rows, `sq` is the chunk's row count and
// the bias is still indexed over the full query range: bias_sq stays at the
// full extent and bias_row_offset locates the chunk inside it. Those two cannot
// be folded into the bias pointer, because the bias plane stride is set by the
// full extent while the score plane stride is set by the chunk -- so with
// bias_batch or bias_heads > 1 a folded pointer would land in the wrong plane.
//
// Each case is run twice: once untiled, and once as a sequence of chunks whose
// results must reproduce the untiled answer row for row. The chunk sizes are
// chosen so the final chunk is ragged, which is what a real prompt length
// produces.
//
// Build (from this directory):
//   hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Wno-deprecated-declarations \
//     test_gqa_attention_bias.cpp ../../../../hip/gqa_kernel.hip \
//     -I../../../../include -o test_gqa_attention_bias.exe
//   ./test_gqa_attention_bias.exe
// ============================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int hip_gqa_add_attention_bias_f32(
    void* stream, void* scores, const void* bias, int total_heads,
    int num_heads, int bias_batch, int bias_heads, int sq, int total_seq,
    int score_batch_stride, int bias_element_size_bytes, int bias_sq,
    int bias_row_offset);

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),        \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Case {
  const char* name;
  int B, num_heads, sq, total_seq;
  int bias_batch;  // 1 to broadcast, else B
  int bias_heads;  // 1 to broadcast, else num_heads
  int elem;        // bias element size: 2 (fp16) or 4 (fp32)
  int chunk;       // query rows per chunk in the tiled arm
  // Pass bias_sq = 0 instead of the real extent in the untiled arm, exercising
  // the kernel's "bias_sq <= 0 means sq" defaulting.
  bool bias_sq_zero;
};

// scores[head][row][col] += bias[bias_b][bias_h][row][col], with the same
// broadcast rules the kernel implements.
static void cpu_reference(std::vector<float>& scores,
                          const std::vector<float>& bias, int total_heads,
                          int num_heads, int bias_batch, int bias_heads, int sq,
                          int total_seq) {
  const size_t bias_plane = (size_t)sq * total_seq;
  for (int head = 0; head < total_heads; ++head) {
    const int b = head / num_heads;
    const int h = head % num_heads;
    const int bb = (bias_batch == 1) ? 0 : b;
    const int bh = (bias_heads == 1) ? 0 : h;
    const size_t bias_base =
        (size_t)bb * bias_heads * bias_plane + (size_t)bh * bias_plane;
    for (int r = 0; r < sq; ++r)
      for (int c = 0; c < total_seq; ++c)
        scores[(size_t)head * sq * total_seq + (size_t)r * total_seq + c] +=
            bias[bias_base + (size_t)r * total_seq + c];
  }
}

static double max_abs_diff(const std::vector<float>& a,
                           const std::vector<float>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::fmax(m, std::fabs((double)a[i] - (double)b[i]));
  return m;
}

static bool run_case(const Case& c) {
  const int total_heads = c.B * c.num_heads;
  const size_t score_n = (size_t)total_heads * c.sq * c.total_seq;
  const size_t bias_n =
      (size_t)c.bias_batch * c.bias_heads * c.sq * c.total_seq;

  std::mt19937 rng(99 + c.sq + c.total_seq + c.elem + c.chunk);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

  std::vector<float> scores0(score_n), bias_f(bias_n);
  for (auto& x : scores0) x = dist(rng);
  for (auto& x : bias_f) x = dist(rng);

  // Round the bias through its storage type so the reference reads exactly what
  // the kernel reads; otherwise the comparison would charge the kernel for the
  // host's fp16 rounding.
  std::vector<__half> bias_h(bias_n);
  if (c.elem == 2) {
    for (size_t i = 0; i < bias_n; ++i) {
      bias_h[i] = __float2half(bias_f[i]);
      bias_f[i] = __half2float(bias_h[i]);
    }
  }

  std::vector<float> expect = scores0;
  cpu_reference(expect, bias_f, total_heads, c.num_heads, c.bias_batch,
                c.bias_heads, c.sq, c.total_seq);

  void* dBias = nullptr;
  const size_t bias_bytes = bias_n * (c.elem == 2 ? sizeof(__half) : sizeof(float));
  HIP_CHECK(hipMalloc(&dBias, bias_bytes));
  HIP_CHECK(hipMemcpy(dBias,
                      c.elem == 2 ? (const void*)bias_h.data()
                                  : (const void*)bias_f.data(),
                      bias_bytes, hipMemcpyHostToDevice));

  // ---- Arm A: untiled, one call over the whole query range ----
  float* dScores = nullptr;
  HIP_CHECK(hipMalloc(&dScores, score_n * sizeof(float)));
  HIP_CHECK(hipMemcpy(dScores, scores0.data(), score_n * sizeof(float),
                      hipMemcpyHostToDevice));
  int rc = hip_gqa_add_attention_bias_f32(
      nullptr, dScores, dBias, total_heads, c.num_heads, c.bias_batch,
      c.bias_heads, c.sq, c.total_seq, c.sq * c.total_seq, c.elem,
      c.bias_sq_zero ? 0 : c.sq, 0);
  HIP_CHECK(hipDeviceSynchronize());
  if (rc != 0) {
    fprintf(stderr, "%s: untiled call returned %d\n", c.name, rc);
    HIP_CHECK(hipFree(dScores));
    HIP_CHECK(hipFree(dBias));
    return false;
  }
  std::vector<float> got_untiled(score_n);
  HIP_CHECK(hipMemcpy(got_untiled.data(), dScores, score_n * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(dScores));

  // ---- Arm B: chunked, mirroring the tiled prefill's call sequence ----
  // Each chunk gets its own [total_heads, rows, total_seq] score buffer with a
  // per-head stride of rows*total_seq, while the bias keeps its full extent and
  // is located by bias_row_offset.
  std::vector<float> got_chunked(score_n);
  bool chunk_ok = true;
  for (int q0 = 0; q0 < c.sq && chunk_ok; q0 += c.chunk) {
    const int rows = std::min(c.chunk, c.sq - q0);
    const size_t chunk_n = (size_t)total_heads * rows * c.total_seq;
    std::vector<float> chunk_in(chunk_n);
    for (int head = 0; head < total_heads; ++head)
      for (int r = 0; r < rows; ++r)
        std::memcpy(&chunk_in[((size_t)head * rows + r) * c.total_seq],
                    &scores0[((size_t)head * c.sq + q0 + r) * c.total_seq],
                    (size_t)c.total_seq * sizeof(float));

    float* dChunk = nullptr;
    HIP_CHECK(hipMalloc(&dChunk, chunk_n * sizeof(float)));
    HIP_CHECK(hipMemcpy(dChunk, chunk_in.data(), chunk_n * sizeof(float),
                        hipMemcpyHostToDevice));
    rc = hip_gqa_add_attention_bias_f32(
        nullptr, dChunk, dBias, total_heads, c.num_heads, c.bias_batch,
        c.bias_heads, rows, c.total_seq, rows * c.total_seq, c.elem, c.sq, q0);
    HIP_CHECK(hipDeviceSynchronize());
    if (rc != 0) {
      fprintf(stderr, "%s: chunk at q0=%d returned %d\n", c.name, q0, rc);
      chunk_ok = false;
    } else {
      std::vector<float> chunk_out(chunk_n);
      HIP_CHECK(hipMemcpy(chunk_out.data(), dChunk, chunk_n * sizeof(float),
                          hipMemcpyDeviceToHost));
      for (int head = 0; head < total_heads; ++head)
        for (int r = 0; r < rows; ++r)
          std::memcpy(&got_chunked[((size_t)head * c.sq + q0 + r) * c.total_seq],
                      &chunk_out[((size_t)head * rows + r) * c.total_seq],
                      (size_t)c.total_seq * sizeof(float));
    }
    HIP_CHECK(hipFree(dChunk));
  }
  HIP_CHECK(hipFree(dBias));
  if (!chunk_ok) return false;

  // Both arms add the same fp32 operands in the same order, so anything other
  // than an exact match is a real addressing error rather than rounding.
  const double d_untiled = max_abs_diff(got_untiled, expect);
  const double d_chunked = max_abs_diff(got_chunked, expect);
  const bool pass = (d_untiled == 0.0) && (d_chunked == 0.0);

  printf("%-18s B%d H%-3d sq=%-5d tseq=%-5d bias[%d,%d] %s chunk=%-5d | "
         "untiled=%.1e chunked=%.1e  %s\n",
         c.name, c.B, c.num_heads, c.sq, c.total_seq, c.bias_batch,
         c.bias_heads, c.elem == 2 ? "fp16" : "fp32", c.chunk, d_untiled,
         d_chunked, pass ? "PASS" : "FAIL");
  return pass;
}

int main() {
  const Case cases[] = {
      // Gemma-4 26B-A4B prefill geometry, which is what the tiled path runs.
      // A fully broadcast bias is the common HF export and cannot catch a plane
      // -stride error, so it is only the baseline.
      {"gemma4-bcast",   1, 16, 1024, 1024, 1, 1, 2, 384, false},
      // bias_heads > 1 is what makes bias_row_offset load-bearing: the plane
      // stride spans the full query extent, so folding the offset into the
      // pointer would read the next head's rows.
      {"gemma4-perhead", 1, 16, 1024, 1024, 1, 16, 2, 384, false},
      // Both batch and head planes real, so an error in either stride shows.
      {"multi-batch",    2,  4,  512,  512, 2,  4, 2, 200, false},
      {"batch-bcast-h",  2,  4,  512,  512, 1,  4, 2, 200, false},
      // fp32 bias takes the other kernel instantiation.
      {"perhead-fp32",   1,  8,  512,  512, 1,  8, 4, 192, false},
      {"multi-batch-f32",2,  4,  256,  256, 2,  4, 4, 100, false},
      // total_seq != sq is the chunked-prefill shape (past_len > 0), where the
      // row and column extents must not be conflated.
      {"past-nonsquare", 1,  8,  512, 1536, 1,  8, 2, 192, false},
      {"past-square-tail",1, 8,  500, 1500, 1,  8, 2, 128, false},
      // Chunk that divides sq exactly: no ragged tail.
      {"exact-chunks",   1,  8,  512,  512, 1,  8, 2, 128, false},
      // One row per chunk is the floor the runtime falls back to when a single
      // row already exceeds the budget.
      {"single-row",     1,  4,   16,   64, 1,  4, 2,   1, false},
      // Chunk wider than sq must behave as a single untiled pass.
      {"chunk-gt-sq",    1,  4,   64,   64, 1,  4, 2, 256, false},
      // bias_sq = 0 must default to sq on the untiled call.
      {"bias-sq-zero",   1,  8,  256,  256, 1,  8, 2, 100, true},
  };

  int fails = 0;
  for (const auto& c : cases)
    if (!run_case(c)) ++fails;
  printf("\n%s (%d failing case(s))\n", fails == 0 ? "ALL PASS" : "SOME FAILED",
         fails);
  return fails == 0 ? 0 : 1;
}
