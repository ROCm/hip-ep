/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- WinogradKernel.cpp - Build hip.winograd_conv kernels ---------------===//
//
// Reproduces what the MIGraphX JIT does for its winograd_conv op
// (src/targets/gpu/jit/winograd_conv.cpp and compile_hip_code_object.cpp at
// the vendored commit): an args.hpp with compile-time lens and strides, the
// kernel template with one configuration, the fused epilogue, and its
// defines. hiprtc compiles it against the embedded headers.
//
//===----------------------------------------------------------------------===//

#include "hip/Compiler/WinogradKernel.h"
#include "hip/Compiler/RocMlirAutotune.h"

#include "hip/Dialect/IR/HipDialect.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/Threading.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#if defined(HIPDNN_HIPRTC_LIB)
#ifdef _WIN32
// Declared rather than including <windows.h>, whose macros collide with the
// MLIR/LLVM headers here. Same pattern as RocMlirKernelCompiler.cpp.
extern "C" __declspec(dllimport) void *__stdcall GetModuleHandleA(const char *);
extern "C" __declspec(dllimport) void *__stdcall LoadLibraryExA(const char *,
                                                                void *,
                                                                unsigned long);
extern "C" __declspec(dllimport) void *__stdcall GetProcAddress(void *,
                                                                const char *);
#define HIPDNN_LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
#define HIPDNN_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x00001000
#else
#include <dlfcn.h>
#endif
#endif

// The vendored MIGraphX kernel headers, embedded by lib/Compiler/CMakeLists.txt
// as NUL-terminated byte arrays.
extern "C" const char *const kWinogradHeaderNames[];
extern "C" const unsigned char *const kWinogradHeaderData[];
extern "C" const size_t kWinogradHeaderCount;

namespace mlir::hip {

bool winogradKernelSupportsArch(llvm::StringRef arch) {
  return arch == "gfx1151" || arch.starts_with("gfx12");
}

WinogradConfig chooseWinogradConfig(int64_t n, int64_t c, int64_t h, int64_t w,
                                    int64_t k) {
  (void)n;
  WinogradConfig config;
  const int64_t spatial = std::min(h, w);
  // From a sweep of the 20 MIGraphX fp16 configurations on gfx1151 over the
  // shapes that the gate admits: fold-at-end with the T store wins most; a
  // narrow output over a large image prefers three output blocks per group
  // and the raw filter; large channel counts prefer eight waves. With
  // autotuning on, compileWinogradKernels times candidates per shape instead.
  if (k <= 32 && spatial >= 256) {
    config.outBlocks = 3;
    config.foldAtEnd = false;
    config.fullTransform = true;
  } else if (std::min(c, k) >= 448) {
    config.waves = 8;
    config.foldAtEnd = false;
    config.fullTransform = true;
  }
  return config;
}

namespace {

std::string ints(llvm::ArrayRef<int64_t> values) {
  std::string s;
  for (int64_t v : values) {
    if (!s.empty())
      s += ",";
    s += std::to_string(v);
  }
  return s;
}

std::vector<int64_t> packedStrides(llvm::ArrayRef<int64_t> lens) {
  std::vector<int64_t> strides(lens.size());
  int64_t acc = 1;
  for (size_t i = lens.size(); i-- > 0;) {
    strides[i] = acc;
    acc *= lens[i];
  }
  return strides;
}

std::string makeTensor(int index, llvm::ArrayRef<int64_t> lens,
                       llvm::ArrayRef<int64_t> strides) {
  return llvm::formatv(
      "template<> struct make_tensor<{0}> {{\n"
      "  static __device__ auto apply(void* __restrict__ p) {{\n"
      "    return make_tensor_view(reinterpret_cast<half* __restrict__>(p),\n"
      "        make_shape(index_ints<{1}>{{}, index_ints<{2}>{{}));\n"
      "  }\n};\n",
      index, ints(lens), ints(strides));
}

} // namespace

WinogradKernelSource
buildWinogradKernelSource(int64_t n, int64_t c, int64_t h, int64_t w, int64_t k,
                          const WinogradConfig &config, bool hasBias,
                          llvm::StringRef activation, double alpha) {
  std::vector<int64_t> x = {n, c, h, w}, y = {n, k, h, w};
  std::vector<int64_t> u = {config.fullTransform ? 3 : 4, 3, k, c};

  // Kernel arguments: input, filter store, [bias], output. rotate_last()
  // hands the output to the kernel first, as in the MIGraphX JIT. The bias is
  // read at every output position through a broadcast view of [K].
  std::string tensors =
      makeTensor(0, x, packedStrides(x)) + makeTensor(1, u, packedStrides(u));
  int nargs = 2;
  if (hasBias)
    tensors += makeTensor(nargs++, y, {0, 1, 0, 0});
  tensors += makeTensor(nargs++, y, packedStrides(y));

  // The epilogue runs on a half or on a vec<half, 8>, so it is written with
  // operations that both support. max(s, s * a) is LeakyRelu for 0 <= a <= 1
  // and ReLU for a = 0.
  std::string value = hasBias ? "v + b" : "v";
  std::string post = "op::id{}", postDef;
  if (hasBias || activation != "none") {
    std::string body = "auto s = " + value + "; ";
    if (activation == "relu")
      body += "return migraphx::max(s, s * static_cast<half>(0.0f));";
    else if (activation == "leaky_relu")
      body += "return migraphx::max(s, s * static_cast<half>(" +
              llvm::formatv("{0:e9}", alpha).str() + "f));";
    else
      body += "return s;";
    postDef = std::string("struct winograd_post {\n  template <class T") +
              (hasBias ? ", class B" : "") +
              "> __device__ auto operator()(T v" + (hasBias ? ", B b" : "") +
              ") const { " + body + " }\n};\n";
    post = "winograd_post{}";
  }

  std::string params, args;
  for (int i = 0; i < nargs; ++i) {
    params += (i ? ", " : "") + std::string("void* p") + std::to_string(i);
    args += (i ? ", " : "") + std::string("p") + std::to_string(i);
  }

  // The kernel loads each input row of a tile with one 4-wide vector load
  // that starts at column -1. Two such loads cross an end of the input
  // buffer: row 0 of image 0, channel 0 (it starts before the buffer) and
  // the last row of the last image and channel (it ends after it). A buffer
  // load that crosses the range reads 0 in all its lanes, so the outputs that
  // use those rows lose their in-range values: the top-left 2x2 tile of image
  // 0 and the bottom-right tiles of the last image. A per-element load
  // instead spills registers and halves the speed, so `<name>_fixup`
  // recomputes those at most 12 positions per output channel directly, with
  // the same epilogue, after the main kernel. The filter comes from the
  // store: g, or T = G g with g0 = T0, g1 = T1 - T2, g2 = T3.
  //
  // The 12 positions read at most 24 distinct input pixels per channel. Each
  // block takes kFixupChannels output channels for all 12 positions: it
  // stages those pixels for a chunk of input channels in LDS, and each wave
  // streams the filter of one output channel once, contiguously, and
  // accumulates all 12 positions. Reading the filter per position instead
  // costs as much as the conv kernel on small images.
  struct Pos {
    int64_t n, h, w;
  };
  std::vector<Pos> positions;
  for (int p = 0; p < 12; ++p)
    positions.push_back(
        p < 4 ? Pos{0, p / 2, p % 2}
              : Pos{n - 1, h - 4 + (p - 4) / 2, w - 2 + (p - 4) % 2});
  std::vector<int64_t> pixels;
  std::string taps, outs;
  for (const Pos &pos : positions) {
    bool valid = pos.h >= 0 && pos.h < h && pos.w >= 0 && pos.w < w;
    outs += (outs.empty() ? "" : ", ") +
            std::to_string(valid ? (pos.n * k * h + pos.h) * w + pos.w : -1);
    std::string row;
    for (int t = 0; t < 9; ++t) {
      int64_t ih = pos.h + t / 3 - 1, iw = pos.w + t % 3 - 1;
      int64_t index = -1;
      if (valid && ih >= 0 && ih < h && iw >= 0 && iw < w) {
        int64_t offset = (pos.n * c * h + ih) * w + iw;
        auto it = llvm::find(pixels, offset);
        index = it - pixels.begin();
        if (it == pixels.end())
          pixels.push_back(offset);
      }
      row += (t ? ", " : "") + std::to_string(index);
    }
    taps += (taps.empty() ? "{" : ", {") + row + "}";
  }
  std::string pixelList;
  for (int64_t offset : pixels)
    pixelList += (pixelList.empty() ? "" : ", ") + std::to_string(offset);

  std::string fixupCall = hasBias                ? "winograd_post{}(v, b[k])"
                          : activation != "none" ? "winograd_post{}(v)"
                                                 : "v";
  std::string filterTap = config.fullTransform
                              ? "at(r, s, c)"
                              : "r == 0 ? at(0, s, c) : r == 2 ? at(3, s, c) : "
                                "at(1, s, c) - at(2, s, c)";
  std::string fixup =
      R"(__global__ void __launch_bounds__(256) @NAME@_fixup(@PARAMS@) {
  constexpr int C = @C@, K = @K@, NP = @NP@;
  constexpr long HW = @HW@;
  constexpr long PIX[NP > 0 ? NP : 1] = {@PIX@};
  constexpr signed char TAP[12][9] = {@TAP@};
  constexpr long OUT[12] = {@OUT@};
  constexpr int KPB = @KPB@, CCH = C < 256 ? C : 256, SLOTS = KPB / 4;
  __shared__ float patch[(NP > 0 ? NP : 1) * CCH];
  const half* x = static_cast<const half*>(p0);
  const half* u = static_cast<const half*>(p1);
  @BIAS@half* y = static_cast<half*>(p@OUTARG@);
  const int waves = blockDim.x / warpSize, wave = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize, k0 = blockIdx.x * KPB;
  float acc[SLOTS][12] = {};
  for (int c0 = 0; c0 < C; c0 += CCH) {
    const int cn = C - c0 < CCH ? C - c0 : CCH;
    // Issue every load of the chunk before the first store: the pixels sit
    // in C different channel planes, so one dependent load at a time costs
    // a full memory latency each.
    constexpr int PER = (NP * CCH + 255) / 256;
    float staged[PER > 0 ? PER : 1];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int i = threadIdx.x + j * 256;
      staged[j] = i < NP * cn
          ? static_cast<float>(x[PIX[i % NP] + (c0 + i / NP) * HW]) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int i = threadIdx.x + j * 256;
      if (i < NP * cn) patch[(i % NP) * CCH + i / NP] = staged[j];
    }
    __syncthreads();
#pragma unroll
    for (int sl = 0; sl < SLOTS; ++sl) {
      const int kl = wave + sl * waves, k = k0 + kl;
      if (kl >= KPB || k >= K) break;
      auto at = [&](int i, int j, int ch) {
        return static_cast<float>(u[((i * 3 + j) * K + k) * static_cast<long>(C) + ch]);
      };
      for (int cl = lane; cl < cn; cl += warpSize) {
        const int c = c0 + cl;
        float g[9];
#pragma unroll
        for (int r = 0; r < 3; ++r)
#pragma unroll
          for (int s = 0; s < 3; ++s)
            g[r * 3 + s] = @FILTER@;
#pragma unroll
        for (int p = 0; p < 12; ++p) {
          if (OUT[p] < 0) continue;
          float a = 0.f;
#pragma unroll
          for (int t = 0; t < 9; ++t)
            if (TAP[p][t] >= 0) a += g[t] * patch[TAP[p][t] * CCH + cl];
          acc[sl][p] += a;
        }
      }
    }
  }
#pragma unroll
  for (int sl = 0; sl < SLOTS; ++sl) {
    const int kl = wave + sl * waves, k = k0 + kl;
    if (kl >= KPB || k >= K) break;
#pragma unroll
    for (int p = 0; p < 12; ++p) {
      if (OUT[p] < 0) continue;
      float sum = acc[sl][p];
      for (int off = warpSize / 2; off > 0; off /= 2)
        sum += __shfl_down(sum, off);
      if (lane == 0) {
        const half v = static_cast<half>(sum);
        y[OUT[p] + k * HW] = @CALL@;
      }
    }
  }
}
)";
  auto fill = [&](llvm::StringRef key, const std::string &value) {
    for (size_t at; (at = fixup.find(key.str())) != std::string::npos;)
      fixup.replace(at, key.size(), value);
  };
  fill("@PARAMS@", params);
  fill("@C@", std::to_string(c));
  fill("@K@", std::to_string(k));
  fill("@NP@", std::to_string(pixels.size()));
  fill("@HW@", std::to_string(h * w));
  fill("@PIX@", pixelList.empty() ? "0" : pixelList);
  fill("@TAP@", taps);
  fill("@OUT@", outs);
  fill("@KPB@", std::to_string(kFixupChannels));
  fill("@BIAS@",
       hasBias ? "const half* b = static_cast<const half*>(p2);\n  " : "");
  fill("@OUTARG@", std::to_string(nargs - 1));
  fill("@FILTER@", filterTap);
  fill("@CALL@", fixupCall);

  WinogradKernelSource out;
  out.name = "hip_winograd_conv_" + activation.str() + (hasBias ? "_bias" : "");
  for (size_t at; (at = fixup.find("@NAME@")) != std::string::npos;)
    fixup.replace(at, 6, out.name);
  out.source =
      llvm::formatv("#include <migraphx/kernels/winograd_conv.hpp>\n"
                    "#include <migraphx/kernels/integral_constant.hpp>\n"
                    "#include <migraphx/kernels/generic_constant.hpp>\n"
                    "#include <migraphx/kernels/ops.hpp>\n"
                    "#include <migraphx/kernels/args.hpp>\n"
                    "#include <migraphx/kernels/tensor_view.hpp>\n"
                    "#include <migraphx/kernels/types.hpp>\n"
                    "namespace migraphx {{\n{0}{1}"
                    "extern \"C\" {{\n"
                    "MIGRAPHX_GLOBAL void {2}({3}) {{\n"
                    "  transform_args(make_tensors(), rotate_last())({4})(\n"
                    "      [](auto output, auto x, auto u, auto... inputs) {{\n"
                    "        winograd_conv_f23_wmma<{5}, {6}, {7}, {8}, {9}, "
                    "{10}, false, half>(\n"
                    "            {11}, output, x, u, inputs...);\n"
                    "      });\n}\n{12}}\n} // namespace migraphx\n",
                    tensors, postDef, out.name, params, args, config.waves,
                    config.channelBlock, config.outBlocks, config.channelSplit,
                    config.foldAtEnd ? "true" : "false",
                    config.fullTransform ? "true" : "false", post, fixup)
          .str();

  // Launch geometry, as in winograd_conv_compiler::compile_op.
  const int64_t tileGroups = 16 * (config.waves / config.channelSplit);
  const int64_t tiles = n * ((h + 1) / 2) * ((w + 1) / 2);
  const int64_t kBlocks =
      (k + 16 * config.outBlocks - 1) / (16 * config.outBlocks);
  const int64_t tBlocks = (tiles + tileGroups - 1) / tileGroups;
  out.blockSize = config.waves * 32;
  out.gridSize = kBlocks * tBlocks;
  return out;
}

namespace {

#if defined(HIPDNN_HIPRTC_LIB)
// hiprtc is resolved at call time, as the HIP runtime is in
// RocMlirKernelCompiler.cpp: a load-time import would stop the compiler
// starting where no ROCm runtime is installed. The prototypes follow
// hip/hiprtc.h; hiprtcResult is an int-sized enum with 0 for success.
using Program = void *;
struct Hiprtc {
  int (*create)(Program *, const char *, const char *, int, const char *const *,
                const char *const *) = nullptr;
  int (*compile)(Program, int, const char *const *) = nullptr;
  int (*logSize)(Program, size_t *) = nullptr;
  int (*log)(Program, char *) = nullptr;
  int (*codeSize)(Program, size_t *) = nullptr;
  int (*code)(Program, char *) = nullptr;
  int (*destroy)(Program *) = nullptr;
  bool ok() const {
    return create && compile && logSize && log && codeSize && code && destroy;
  }
};

const Hiprtc &loadHiprtc() {
  static const Hiprtc api = [] {
    Hiprtc h;
#ifdef _WIN32
    void *mod = GetModuleHandleA(HIPDNN_HIPRTC_LIB);
    if (!mod)
      mod = LoadLibraryExA(HIPDNN_HIPRTC_LIB, nullptr,
                           HIPDNN_LOAD_LIBRARY_SEARCH_SYSTEM32 |
                               HIPDNN_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    auto sym = [&](const char *name) { return GetProcAddress(mod, name); };
#else
    void *mod = dlopen(HIPDNN_HIPRTC_LIB, RTLD_LAZY | RTLD_NOLOAD);
    if (!mod)
      mod = dlopen(HIPDNN_HIPRTC_LIB, RTLD_LAZY);
    auto sym = [&](const char *name) { return dlsym(mod, name); };
#endif
    if (!mod)
      return h;
    h.create = reinterpret_cast<decltype(h.create)>(sym("hiprtcCreateProgram"));
    h.compile =
        reinterpret_cast<decltype(h.compile)>(sym("hiprtcCompileProgram"));
    h.logSize =
        reinterpret_cast<decltype(h.logSize)>(sym("hiprtcGetProgramLogSize"));
    h.log = reinterpret_cast<decltype(h.log)>(sym("hiprtcGetProgramLog"));
    h.codeSize =
        reinterpret_cast<decltype(h.codeSize)>(sym("hiprtcGetCodeSize"));
    h.code = reinterpret_cast<decltype(h.code)>(sym("hiprtcGetCode"));
    h.destroy =
        reinterpret_cast<decltype(h.destroy)>(sym("hiprtcDestroyProgram"));
    return h;
  }();
  return api;
}
#endif

/// Compile `src` for `arch`; the code object on success, the error otherwise.
FailureOr<std::string> compileSource(const WinogradKernelSource &src,
                                     llvm::StringRef arch, std::string &error) {
#if defined(HIPDNN_HIPRTC_LIB)
  const Hiprtc &rtc = loadHiprtc();
  if (!rtc.ok()) {
    error = "hiprtc (" HIPDNN_HIPRTC_LIB ") is not available";
    return failure();
  }
  Program prog = nullptr;
  if (rtc.create(&prog, src.source.c_str(), "winograd_conv.cpp",
                 static_cast<int>(kWinogradHeaderCount),
                 reinterpret_cast<const char *const *>(kWinogradHeaderData),
                 kWinogradHeaderNames) != 0) {
    error = "hiprtcCreateProgram failed";
    return failure();
  }
  std::vector<std::string> options = {
      "-O3",
      "-std=c++17",
      "-fno-gpu-rdc",
      "--offload-arch=" + arch.str(),
      "-DMIGRAPHX_USE_HIPRTC=1",
      "-DMIGRAPHX_NGLOBAL=" + std::to_string(src.gridSize * src.blockSize),
      "-DMIGRAPHX_NLOCAL=" + std::to_string(src.blockSize),
      "-DMIGRAPHX_WAVEFRONTSIZE=32",
      "-Wno-unused-command-line-argument"};
  std::vector<const char *> cOptions;
  for (const std::string &o : options)
    cOptions.push_back(o.c_str());
  int status =
      rtc.compile(prog, static_cast<int>(cOptions.size()), cOptions.data());
  std::string code;
  if (status != 0) {
    size_t size = 0;
    rtc.logSize(prog, &size);
    std::string log(size, '\0');
    if (size)
      rtc.log(prog, log.data());
    error = "hiprtc compile failed: " + log;
  } else {
    size_t size = 0;
    rtc.codeSize(prog, &size);
    code.resize(size);
    if (size)
      rtc.code(prog, code.data());
    if (code.empty())
      error = "hiprtc produced an empty code object";
  }
  rtc.destroy(&prog);
  if (code.empty())
    return failure();
  return code;
#else
  (void)src;
  (void)arch;
  error = "this build has no hiprtc";
  return failure();
#endif
}

} // namespace

namespace {

// MIGraphX fp16 configurations (winograd_conv_compiler::get_tuning_config)
// timed per shape when autotuning is on: the defaults, eight waves, channel
// splits across two waves, fewer waves, and three output blocks.
WinogradConfig candidate(int64_t waves, int64_t outBlocks, int64_t split,
                         bool foldAtEnd, bool fullTransform) {
  WinogradConfig c;
  c.waves = waves;
  c.outBlocks = outBlocks;
  c.channelSplit = split;
  c.foldAtEnd = foldAtEnd;
  c.fullTransform = fullTransform;
  return c;
}

std::vector<WinogradConfig> tuningCandidates(int64_t n, int64_t c, int64_t h,
                                             int64_t w, int64_t k) {
  std::vector<WinogradConfig> out = {
      chooseWinogradConfig(n, c, h, w, k), candidate(4, 1, 1, true, false),
      candidate(8, 1, 1, false, true),     candidate(8, 1, 1, true, false),
      candidate(2, 1, 1, true, false),     candidate(2, 1, 2, false, true),
      candidate(4, 1, 2, false, true),     candidate(4, 3, 1, false, true)};
  auto same = [](const WinogradConfig &a, const WinogradConfig &b) {
    return a.waves == b.waves && a.outBlocks == b.outBlocks &&
           a.channelSplit == b.channelSplit && a.foldAtEnd == b.foldAtEnd &&
           a.fullTransform == b.fullTransform;
  };
  std::vector<WinogradConfig> unique;
  for (const WinogradConfig &cfg : out)
    if (llvm::none_of(unique,
                      [&](const WinogradConfig &u) { return same(u, cfg); }))
      unique.push_back(cfg);
  return unique;
}

std::string configName(const WinogradConfig &c) {
  return llvm::formatv("nw={0} kw={1} sk={2} fe={3} ft={4}", c.waves,
                       c.outBlocks, c.channelSplit, c.foldAtEnd ? 1 : 0,
                       c.fullTransform ? 1 : 0)
      .str();
}

} // namespace

LogicalResult compileWinogradKernels(ModuleOp module, llvm::StringRef arch,
                                     llvm::raw_ostream *log) {
  struct Kernel {
    int64_t n, c, h, w, k;
    bool hasBias;
    std::string act;
    double alpha;
    WinogradConfig config;
    WinogradKernelSource src;
    std::string binary, error;
  };
  llvm::StringMap<size_t> byKey;
  std::vector<Kernel> kernels;
  std::vector<std::pair<WinogradConvOp, size_t>> uses;
  module.walk([&](WinogradConvOp op) {
    auto x = cast<ShapedType>(op.getInput().getType()).getShape();
    auto wt = cast<ShapedType>(op.getWeights().getType()).getShape();
    Kernel kn{x[0],
              x[1],
              x[2],
              x[3],
              wt[0],
              static_cast<bool>(op.getBias()),
              op.getActivation().str(),
              op.getAlpha().convertToDouble(),
              {},
              {},
              {},
              {}};
    std::string key =
        llvm::formatv("{0}x{1}x{2}x{3}->{4} {5} {6} {7}", kn.n, kn.c, kn.h,
                      kn.w, kn.k, kn.hasBias, kn.act, kn.alpha)
            .str();
    auto [it, inserted] = byKey.try_emplace(key, kernels.size());
    if (inserted)
      kernels.push_back(std::move(kn));
    uses.push_back({op, it->second});
  });
  if (kernels.empty())
    return success();

  auto compileAll = [&](std::vector<Kernel> &list) {
    llvm::DefaultThreadPool pool(llvm::hardware_concurrency());
    for (Kernel &kn : list)
      pool.async([&kn, arch] {
        kn.src =
            buildWinogradKernelSource(kn.n, kn.c, kn.h, kn.w, kn.k, kn.config,
                                      kn.hasBias, kn.act, kn.alpha);
        FailureOr<std::string> binary = compileSource(kn.src, arch, kn.error);
        if (succeeded(binary))
          kn.binary = std::move(*binary);
      });
    pool.wait();
  };

  // With autotuning on, time the candidate configurations of each distinct
  // shape on the device (as the MIGraphX JIT does) and keep the fastest. The
  // epilogue barely moves the time, so each shape is timed once, with the
  // epilogue of its first use. Without a device the rule's choice stands.
  for (Kernel &kn : kernels)
    kn.config = chooseWinogradConfig(kn.n, kn.c, kn.h, kn.w, kn.k);
  AutotuneSpace space;
  if (rocMlirAutotuneFromEnv(space)) {
    llvm::StringMap<size_t> shapeOf;
    std::vector<Kernel> trials;
    std::vector<size_t> trialShape;
    std::vector<const Kernel *> shapes;
    for (const Kernel &kn : kernels) {
      std::string shape =
          llvm::formatv("{0}x{1}x{2}x{3}->{4}", kn.n, kn.c, kn.h, kn.w, kn.k)
              .str();
      if (!shapeOf.try_emplace(shape, shapes.size()).second)
        continue;
      for (const WinogradConfig &cfg :
           tuningCandidates(kn.n, kn.c, kn.h, kn.w, kn.k)) {
        Kernel trial = kn;
        trial.config = cfg;
        trials.push_back(std::move(trial));
        trialShape.push_back(shapes.size());
      }
      shapes.push_back(&kn);
    }
    compileAll(trials);
    std::vector<std::optional<double>> best(shapes.size());
    std::vector<WinogradConfig> bestConfig(shapes.size());
    for (size_t i = 0; i < trials.size(); ++i) {
      const Kernel &tr = trials[i];
      if (tr.binary.empty())
        continue;
      std::vector<size_t> bytes = {
          static_cast<size_t>(tr.n * tr.c * tr.h * tr.w * 2),
          static_cast<size_t>((tr.config.fullTransform ? 9 : 12) * tr.k * tr.c *
                              2)};
      if (tr.hasBias)
        bytes.push_back(static_cast<size_t>(tr.k * 2));
      bytes.push_back(static_cast<size_t>(tr.n * tr.k * tr.h * tr.w * 2));
      std::optional<double> ms = benchmarkKernel(
          tr.binary, tr.src.name, tr.src.gridSize, tr.src.blockSize, bytes);
      size_t s = trialShape[i];
      if (log)
        *log << "[winograd] tune " << tr.n << "x" << tr.c << "x" << tr.h << "x"
             << tr.w << "->" << tr.k << " " << configName(tr.config) << ": "
             << (ms ? llvm::formatv("{0:f4} ms", *ms).str() : "n/a") << "\n";
      if (ms && (!best[s] || *ms < *best[s])) {
        best[s] = ms;
        bestConfig[s] = tr.config;
      }
    }
    for (Kernel &kn : kernels) {
      std::string shape =
          llvm::formatv("{0}x{1}x{2}x{3}->{4}", kn.n, kn.c, kn.h, kn.w, kn.k)
              .str();
      size_t s = shapeOf.lookup(shape);
      if (best[s])
        kn.config = bestConfig[s];
    }
  }

  compileAll(kernels);
  for (auto &[op, index] : uses) {
    const Kernel &kn = kernels[index];
    if (kn.binary.empty()) {
      op.emitError() << "Winograd kernel build failed for " << kn.n << "x"
                     << kn.c << "x" << kn.h << "x" << kn.w << "->" << kn.k
                     << ": " << kn.error;
      return failure();
    }
  }
  MLIRContext *ctx = module.getContext();
  for (const Kernel &kn : kernels)
    if (log)
      *log << "[winograd] built " << kn.src.name << " for " << kn.n << "x"
           << kn.c << "x" << kn.h << "x" << kn.w << "->" << kn.k << " "
           << (kn.hasBias ? "true" : "false") << " " << kn.act << " "
           << llvm::formatv("{0:f2}", kn.alpha) << ": " << kn.binary.size()
           << " bytes, grid " << kn.src.gridSize << " block "
           << kn.src.blockSize << ", " << configName(kn.config) << "\n";
  for (auto &[op, index] : uses) {
    const Kernel &kn = kernels[index];
    WinogradConvOp conv = op;
    conv.setKernelBinaryAttr(StringAttr::get(ctx, kn.binary));
    conv.setKernelNameAttr(StringAttr::get(ctx, kn.src.name));
    conv.setGridSizeAttr(
        IntegerAttr::get(IntegerType::get(ctx, 64), kn.src.gridSize));
    conv.setBlockSizeAttr(
        IntegerAttr::get(IntegerType::get(ctx, 64), kn.src.blockSize));
    conv.setFullTransformAttr(BoolAttr::get(ctx, kn.config.fullTransform));
  }
  return success();
}

} // namespace mlir::hip
