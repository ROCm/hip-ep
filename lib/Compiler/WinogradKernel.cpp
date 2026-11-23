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

#include "hip/Dialect/IR/HipDialect.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <algorithm>
#include <cstddef>
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
  // and the raw filter; large channel counts prefer eight waves.
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
  std::string fixupParams = params;
  std::string fixupCall = hasBias                ? "winograd_post{}(v, b[k])"
                          : activation != "none" ? "winograd_post{}(v)"
                                                 : "v";
  std::string fixup =
      llvm::formatv(
          "__global__ void __launch_bounds__(256) {0}_fixup({1}) {{\n"
          "  constexpr int N = {2}, C = {3}, H = {4}, W = {5}, K = {6};\n"
          "  const half* x = static_cast<const half*>(p0);\n"
          "  const half* u = static_cast<const half*>(p1);\n"
          "  {7}"
          "  half* y = static_cast<half*>(p{8});\n"
          "  const int t = blockIdx.x * blockDim.x + threadIdx.x;\n"
          "  const int pos = t / K, k = t % K;\n"
          "  if (pos >= 12) return;\n"
          "  int n, h, w;\n"
          "  if (pos < 4) {{ n = 0; h = pos / 2; w = pos % 2; }\n"
          "  else {{ n = N - 1; h = H - 4 + (pos - 4) / 2; w = W - 2 + (pos - "
          "4) % 2; }\n"
          "  if (h < 0 || h >= H || w < 0 || w >= W) return;\n"
          "  auto at = [&](int i, int j, int c) {{\n"
          "    return static_cast<float>(u[((i * 3 + j) * K + k) * C + c]); "
          "};\n"
          "  float acc = 0.f;\n"
          "  for (int c = 0; c < C; ++c)\n"
          "    for (int r = 0; r < 3; ++r) {{\n"
          "      const int ih = h + r - 1;\n"
          "      if (ih < 0 || ih >= H) continue;\n"
          "      for (int s = 0; s < 3; ++s) {{\n"
          "        const int iw = w + s - 1;\n"
          "        if (iw < 0 || iw >= W) continue;\n"
          "        const float g = {9};\n"
          "        acc += g * static_cast<float>(x[((static_cast<long>(n) * C "
          "+ c) * H + ih) * W + iw]);\n"
          "      }\n"
          "    }\n"
          "  const half v = static_cast<half>(acc);\n"
          "  y[((static_cast<long>(n) * K + k) * H + h) * W + w] = {10};\n"
          "}\n",
          "{NAME}", fixupParams, n, c, h, w, k,
          hasBias ? "const half* b = static_cast<const half*>(p2);\n  " : "",
          nargs - 1,
          config.fullTransform ? "at(r, s, c)"
                               : "r == 0 ? at(0, s, c) : r == 2 ? at(3, s, c) "
                                 ": at(1, s, c) - at(2, s, c)",
          fixupCall)
          .str();

  WinogradKernelSource out;
  out.name = "hip_winograd_conv_" + activation.str() + (hasBias ? "_bias" : "");
  size_t at = fixup.find("{NAME}");
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

LogicalResult compileWinogradKernels(ModuleOp module, llvm::StringRef arch,
                                     llvm::raw_ostream *log) {
  struct Built {
    std::string binary;
    WinogradKernelSource src;
    bool fullTransform;
  };
  llvm::StringMap<Built> cache;
  MLIRContext *ctx = module.getContext();
  WalkResult walked = module.walk([&](WinogradConvOp op) {
    auto x = cast<ShapedType>(op.getInput().getType()).getShape();
    auto wt = cast<ShapedType>(op.getWeights().getType()).getShape();
    int64_t n = x[0], c = x[1], h = x[2], w = x[3], k = wt[0];
    bool hasBias = static_cast<bool>(op.getBias());
    StringRef act = op.getActivation();
    double alpha = op.getAlpha().convertToDouble();
    std::string key = llvm::formatv("{0}x{1}x{2}x{3}->{4} {5} {6} {7}", n, c, h,
                                    w, k, hasBias, act, alpha)
                          .str();
    auto it = cache.find(key);
    if (it == cache.end()) {
      WinogradConfig config = chooseWinogradConfig(n, c, h, w, k);
      WinogradKernelSource src =
          buildWinogradKernelSource(n, c, h, w, k, config, hasBias, act, alpha);
      std::string error;
      FailureOr<std::string> binary = compileSource(src, arch, error);
      if (failed(binary)) {
        op.emitError() << "Winograd kernel build failed for " << key << ": "
                       << error;
        return WalkResult::interrupt();
      }
      if (log)
        *log << "[winograd] built " << src.name << " for " << key << ": "
             << binary->size() << " bytes, grid " << src.gridSize << " block "
             << src.blockSize << "\n";
      it = cache
               .insert({key, Built{std::move(*binary), std::move(src),
                                   config.fullTransform}})
               .first;
    }
    const Built &b = it->second;
    op.setKernelBinaryAttr(StringAttr::get(ctx, b.binary));
    op.setKernelNameAttr(StringAttr::get(ctx, b.src.name));
    op.setGridSizeAttr(
        IntegerAttr::get(IntegerType::get(ctx, 64), b.src.gridSize));
    op.setBlockSizeAttr(
        IntegerAttr::get(IntegerType::get(ctx, 64), b.src.blockSize));
    op.setFullTransformAttr(BoolAttr::get(ctx, b.fullTransform));
    return WalkResult::advance();
  });
  return failure(walked.wasInterrupted());
}

} // namespace mlir::hip
