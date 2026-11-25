/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Per-shape tile sweep for hip-ep's conv kernel (scratch tool, not part of hip-ep).
// Includes conv_kernel.hip (CONV_SRC) so every kernel template is visible, forces each candidate configuration, checks
// it against conv_direct_kernel (the one-thread-per-output kernel in the same file) and times it with hipEvents
// (median over `rounds` interleaved rounds of `iters` back-to-back launches). "default" is hip_conv itself (that build's ladder).
// usage: sweep.exe <shapes.txt> <out.csv> [--batches 1,8] [--configs a,b,...] [--iters 20] [--dtype f16|f32|bf16]
// shapes.txt lines: id cin h w cout g k0 k1 s0 s1 p0 p1 d0 d1 oh ow

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

struct Shape {
  std::string id;
  int64_t cin, h, w, cout, g, k0, k1, s0, s1, p0, p1, d0, d1, oh, ow;
};

static bool rank1(const Shape &s) { return s.w == 1 && s.k1 == 1 && s.ow == 1 && s.p1 == 0; }

static ConvParams makeParams(const Shape &s, int64_t N) {
  ConvParams pr{};
  pr.N = N; pr.Cin = s.cin; pr.Cout = s.cout; pr.group = s.g;
  pr.CinPerGroup = s.cin / s.g; pr.MPerGroup = s.cout / s.g;
  pr.in0 = s.h; pr.in1 = s.w; pr.in2 = 1;
  pr.in0i = (int32_t)s.h; pr.in1i = (int32_t)s.w; pr.in2i = 1;
  pr.out0 = s.oh; pr.out1 = s.ow; pr.out2 = 1;
  pr.k0 = s.k0; pr.k1 = s.k1; pr.k2 = 1; pr.s0 = s.s0; pr.s1 = s.s1; pr.s2 = 1;
  pr.p0 = s.p0; pr.p1 = s.p1; pr.p2 = 0; pr.d0 = s.d0; pr.d1 = s.d1; pr.d2 = 1;
  pr.kvol = s.k0 * s.k1; pr.K = pr.CinPerGroup * pr.kvol; pr.Nout = s.oh * s.ow;
  pr.inPlane = s.h * s.w; pr.k1k2 = s.k1; pr.out1out2 = s.ow;
  pr.rKvol = makeFastDiv(pr.kvol); pr.rK1k2 = makeFastDiv(pr.k1k2); pr.rK2 = makeFastDiv(1);
  pr.rOut1out2 = makeFastDiv(pr.out1out2); pr.rOut2 = makeFastDiv(1);
  pr.innerAxis = 1; pr.innerIni = (int32_t)s.w; pr.innerOut = s.ow; pr.rInnerOut = makeFastDiv(s.ow);
  pr.innerVec = (s.s1 == 1 && s.ow >= 8) ? 1 : 0;
  if (rank1(s)) {  // a 1-D conv, laid out as the caller lays it out: axis 0 is the only spatial axis
    pr.innerAxis = 0; pr.innerIni = (int32_t)s.h; pr.innerOut = s.oh; pr.rInnerOut = makeFastDiv(s.oh);
    pr.innerVec = (s.s0 == 1 && s.oh >= 8) ? 1 : 0;
  }
  return pr;
}

using Launch = std::function<void(hipStream_t, const void *, const void *, const void *, void *, const ConvParams &)>;
struct Cfg {
  std::string name;
  bool window;
  Launch fn;
  std::function<int64_t(const ConvParams &)> blocks;
};

static int64_t tileBlocks(const ConvParams &p, int64_t bm, int64_t bn) {
  return ((p.MPerGroup + bm - 1) / bm) * ((p.Nout + bn - 1) / bn) * p.N * p.group;
}
static int64_t winBlocks(const ConvParams &p, int64_t bm, int64_t th, int64_t tw) {
  return ((p.MPerGroup + bm - 1) / bm) * ((p.out0 + th - 1) / th) * ((p.out1 + tw - 1) / tw) * p.N * p.group;
}

#define WMMA(BM, BN, WM, WN)                                                                              \
  Cfg{"wmma" #BM "x" #BN, false,                                                                          \
      [](hipStream_t s, const void *x, const void *w, const void *b, void *y, const ConvParams &p) {    \
        launchWmma<BM, BN, WM, WN>(s, x, w, b, y, p);                                                     \
      },                                                                                                  \
      [](const ConvParams &p) { return tileBlocks(p, BM, BN); }}
#define WIN(BM, TH, TW, WM, WN)                                                                           \
  Cfg{"win" #BM "_" #TH "x" #TW, true,                                                                    \
      [](hipStream_t s, const void *x, const void *w, const void *b, void *y, const ConvParams &p) {    \
        launchWindow<BM, TH, TW, WM, WN>(s, x, w, b, y, p);                                               \
      },                                                                                                  \
      [](const ConvParams &p) { return winBlocks(p, BM, TH, TW); }}

#ifndef EXTRA_CFGS
#define EXTRA_CFGS
#endif

static std::vector<Cfg> allCfgs() {
  return {WMMA(128, 256, 4, 4), WMMA(256, 128, 4, 2), WMMA(128, 128, 4, 2), WMMA(64, 256, 2, 4),
          WMMA(32, 256, 2, 2),  WMMA(16, 256, 1, 2),  WMMA(128, 64, 4, 1),  WMMA(64, 128, 2, 2),
          WMMA(64, 64, 2, 2),   WMMA(32, 128, 2, 2),  WMMA(32, 64, 1, 2),   WMMA(16, 128, 1, 2),
          WMMA(16, 64, 1, 1),   WMMA(64, 32, 2, 1),   WMMA(32, 32, 1, 1),   WIN(64, 16, 16, 2, 4),
          WIN(32, 8, 32, 2, 2), WIN(64, 8, 16, 2, 2), WIN(32, 8, 16, 2, 2), WIN(64, 4, 16, 2, 2),
          WIN(32, 4, 16, 1, 2), WIN(16, 8, 16, 1, 2),
          Cfg{"dwkernel", false,
              [](hipStream_t s, const void *x, const void *w, const void *b, void *y, const ConvParams &p) {
                launchDepthwiseForShape<__half>(s, x, w, b, y, p);
              },
              [](const ConvParams &p) { return p.N * p.Cout; }} EXTRA_CFGS};
}

template <typename T> static T fromF(float v);
template <> __half fromF<__half>(float v) { return __float2half(v); }
template <> float fromF<float>(float v) { return v; }
template <> hip_bfloat16 fromF<hip_bfloat16>(float v) { return hip_bfloat16(v); }
template <typename T> static float toF(T v);
template <> float toF<__half>(__half v) { return __half2float(v); }
template <> float toF<float>(float v) { return v; }
template <> float toF<hip_bfloat16>(hip_bfloat16 v) { return (float)v; }

template <typename T>
static int runAll(const std::vector<Shape> &shapes, const std::vector<int> &batches,
                  const std::vector<std::string> &want, const std::vector<int> &splitList, int iters, int rounds, FILE *out, int dtypeCode, const char *dtName) {
  hipStream_t st;
  CK(hipStreamCreate(&st));
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  auto cfgs = allCfgs();
  auto wanted = [&](const std::string &n) { return want.empty() || std::find(want.begin(), want.end(), n) != want.end(); };
  for (const auto &s : shapes)
    for (int B : batches) {
      ConvParams pr = makeParams(s, B);
      const size_t nx = B * s.cin * s.h * s.w, nw = s.cout * pr.CinPerGroup * pr.kvol, ny = B * s.cout * pr.Nout;
      std::vector<T> hx(nx), hw(nw), hb(s.cout);
      const float ws = 1.0f / std::sqrt((float)pr.K);
      for (auto &v : hx) v = fromF<T>(U(rng));
      for (auto &v : hw) v = fromF<T>(U(rng) * ws);
      for (auto &v : hb) v = fromF<T>(U(rng) * 0.1f);
      T *dx, *dw, *db, *dy, *dref;
      CK(hipMalloc(&dx, nx * sizeof(T)));
      CK(hipMalloc(&dw, nw * sizeof(T)));
      CK(hipMalloc(&db, s.cout * sizeof(T)));
      CK(hipMalloc(&dy, ny * sizeof(T)));
      CK(hipMalloc(&dref, ny * sizeof(T)));
      CK(hipMemcpy(dx, hx.data(), nx * sizeof(T), hipMemcpyHostToDevice));
      CK(hipMemcpy(dw, hw.data(), nw * sizeof(T), hipMemcpyHostToDevice));
      CK(hipMemcpy(db, hb.data(), s.cout * sizeof(T), hipMemcpyHostToDevice));
      {
        const int64_t total = (int64_t)ny;
        hipLaunchKernelGGL(conv_direct_kernel<T>, dim3((unsigned)((total + 255) / 256)), dim3(256), 0, st, dx, dw, db,
                           dref, pr, total);
        CK(hipStreamSynchronize(st));
      }
      std::vector<T> href(ny), hy(ny);
      CK(hipMemcpy(href.data(), dref, ny * sizeof(T), hipMemcpyDeviceToHost));
      double refmax = 0;
      for (auto v : href) refmax = std::max(refmax, (double)std::fabs(toF<T>(v)));
      // Correctness first (one launch per config against the reference), then the timing rounds. The rounds are
      // interleaved across configs with alternating order: the part is power-limited and a long burst of one config
      // runs measurably slower than the same kernel measured fresh, so timing configs one after another would favour
      // whichever ran first.
      auto check = [&](const std::function<void()> &launchFn, double &err) -> bool {
        CK(hipMemset(dy, 0, ny * sizeof(T)));
        (void)hipGetLastError();
        launchFn();
        if (hipGetLastError() != hipSuccess || hipStreamSynchronize(st) != hipSuccess) return false;
        CK(hipMemcpy(hy.data(), dy, ny * sizeof(T), hipMemcpyDeviceToHost));
        double e = 0;
        for (size_t i = 0; i < ny; ++i) e = std::max(e, (double)std::fabs(toF<T>(hy[i]) - toF<T>(href[i])));
        err = e / (refmax + 1e-30);
        return true;
      };
      auto batchTime = [&](const std::function<void()> &launchFn) {
        for (int i = 0; i < 2; ++i) launchFn();
        CK(hipEventRecord(e0, st));
        for (int i = 0; i < iters; ++i) launchFn();
        CK(hipEventRecord(e1, st));
        CK(hipEventSynchronize(e1));
        float ms;
        CK(hipEventElapsedTime(&ms, e0, e1));
        return ms * 1000.0 / iters;
      };
      std::vector<std::pair<std::string, std::function<void()>>> todo;
      std::map<std::string, int64_t> blk;
      if (wanted("default"))
        todo.push_back({"default", [&] {
                          hip_conv(st, dx, dw, db, dy, dtypeCode, rank1(s) ? 1 : 2, B, s.cin, s.cout, s.h, s.w, 1, s.oh, s.ow, 1, s.k0,
                                   s.k1, 1, s.s0, s.s1, 1, s.p0, s.p1, 0, s.d0, s.d1, 1, s.g);
                        }});
      float *wsp = nullptr;
#ifdef SWEEP_SPLITK
      size_t wsBytes = 0;
      for (int k : splitList) wsBytes = std::max(wsBytes, (size_t)k * ny * sizeof(float));
      wsBytes = std::max(wsBytes, hip_conv_workspace_size(dtypeCode, rank1(s) ? 1 : 2, B, s.cin, s.cout, s.h, s.w, 1, s.oh,
                                                          s.ow, 1, s.k0, s.k1, 1, s.s0, s.s1, 1, s.p0, s.p1, 0, s.d0,
                                                          s.d1, 1, s.g));
      if (wsBytes) CK(hipMalloc(&wsp, wsBytes));
      if (wanted("auto"))
        todo.push_back({"auto", [&] {
                          hip_conv_ws(st, dx, dw, db, dy, dtypeCode, rank1(s) ? 1 : 2, B, s.cin, s.cout, s.h, s.w, 1,
                                      s.oh, s.ow, 1, s.k0, s.k1, 1, s.s0, s.s1, 1, s.p0, s.p1, 0, s.d0, s.d1, 1, s.g,
                                      wsp, wsBytes);
                        }});
      if (dtypeCode == HIP_DTYPE_FLOAT16)
        for (auto &c : cfgs)
          for (int k : splitList) {
            if (k <= 1) continue;
            const std::string nm = c.name + "/s" + std::to_string(k);
            if (!wanted(nm)) continue;
            if (c.window && !(windowEligible(pr) && pr.CinPerGroup >= kWmma)) continue;
            const int32_t ke = evenSplits(pr, c.window, k);
            if (ke != k) continue;  // only splits that divide the contraction evenly
            const ConvParams ps = withSplit(pr, ke, c.window, wsp);
            blk[nm] = c.blocks(pr) * ke;
            Launch fn = c.fn;
            todo.push_back({nm, [&, fn, ps] {
                              fn(st, dx, dw, db, dy, ps);
                              launchSplitReduce<__half>(st, db, dy, ps);
                            }});
          }
#endif
      if (dtypeCode == HIP_DTYPE_FLOAT16)
        for (auto &c : cfgs) {
          if (!wanted(c.name)) continue;
          if (c.window && !(windowEligible(pr) && pr.CinPerGroup >= kWmma)) continue;
          if (c.name == "dwkernel" && !(pr.CinPerGroup == 1 && pr.kvol <= kMaxDepthwiseK)) continue;
          blk[c.name] = c.blocks(pr);
          Launch fn = c.fn;
          todo.push_back({c.name, [&, fn] { fn(st, dx, dw, db, dy, pr); }});
        }
      std::vector<double> errs(todo.size(), -1);
      std::vector<bool> oks(todo.size());
      std::vector<std::vector<double>> times(todo.size());
      for (size_t i = 0; i < todo.size(); ++i) oks[i] = check(todo[i].second, errs[i]);
      for (int r = 0; r < rounds; ++r)
        for (size_t j = 0; j < todo.size(); ++j) {
          const size_t i = (r % 2) ? todo.size() - 1 - j : j;
          if (oks[i]) times[i].push_back(batchTime(todo[i].second));
        }
      for (size_t i = 0; i < todo.size(); ++i) {
        double us = -1;
        if (!times[i].empty()) {
          std::sort(times[i].begin(), times[i].end());
          us = times[i][times[i].size() / 2];
        }
        fprintf(out, "%s,%s,%d,%s,%lld,%.3f,%.3g,%s\n", s.id.c_str(), dtName, B, todo[i].first.c_str(),
                (long long)(blk.count(todo[i].first) ? blk[todo[i].first] : -1), us, errs[i],
                oks[i] ? (errs[i] < 2e-2 ? "OK" : "BAD") : "FAIL");
      }
      fflush(out);
      CK(hipFree(dx)); CK(hipFree(dw)); CK(hipFree(db)); CK(hipFree(dy)); CK(hipFree(dref));
      if (wsp) CK(hipFree(wsp));
    }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: sweep <shapes.txt> <out.csv> [--batches 1,8] [--configs a,b] [--iters N] [--dtype f16]\n");
    return 1;
  }
  std::vector<int> batches{1, 8};
  std::vector<std::string> want;
  int iters = 10, rounds = 7;
  std::vector<int> splitList;
  std::string dt = "f16";
  auto split = [](const std::string &v) {
    std::vector<std::string> r;
    std::stringstream ss(v);
    std::string t;
    while (std::getline(ss, t, ',')) r.push_back(t);
    return r;
  };
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--batches") {
      batches.clear();
      for (auto &t : split(argv[++i])) batches.push_back(std::stoi(t));
    } else if (a == "--configs") want = split(argv[++i]);
    else if (a == "--iters") iters = std::stoi(argv[++i]);
    else if (a == "--rounds") rounds = std::stoi(argv[++i]);
    else if (a == "--splits") { for (auto &t : split(argv[++i])) splitList.push_back(std::stoi(t)); }
    else if (a == "--dtype") dt = argv[++i];
  }
  std::vector<Shape> shapes;
  FILE *f = fopen(argv[1], "r");
  if (!f) return 1;
  char line[512];
  while (fgets(line, sizeof line, f)) {
    if (line[0] == '#' || strlen(line) < 5) continue;
    std::stringstream ss(line);
    Shape s;
    ss >> s.id >> s.cin >> s.h >> s.w >> s.cout >> s.g >> s.k0 >> s.k1 >> s.s0 >> s.s1 >> s.p0 >> s.p1 >> s.d0 >> s.d1 >>
        s.oh >> s.ow;
    shapes.push_back(s);
  }
  fclose(f);
  FILE *out = fopen(argv[2], "a");
  if (dt == "f16") return runAll<__half>(shapes, batches, want, splitList, iters, rounds, out, HIP_DTYPE_FLOAT16, "f16");
  if (dt == "f32") return runAll<float>(shapes, batches, want, splitList, iters, rounds, out, HIP_DTYPE_FLOAT32, "f32");
  return runAll<hip_bfloat16>(shapes, batches, want, splitList, iters, rounds, out, HIP_DTYPE_BFLOAT16, "bf16");
}
