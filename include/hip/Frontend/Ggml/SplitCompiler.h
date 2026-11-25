/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- SplitCompiler.h - Compiling a ggml backend's split -----------------===//
//
// A split (SplitGraph.h) captured and compiled, in libhipep_backend's process
// (lib/HipepBackend): this frontend and hip-ep's compiler.
//
//   capture = captureSplit(split, tensors, symbols);
//     - the split's nodes and what they read, written down; the capture
//       refers to no tensor after it returns.
//   compileSplit(*capture, directory);
//     - the code, in `directory`.
//
// `symbols` is the SymbolSpec the code is compiled with - empty for code of
// the split's sizes alone - or null: they are derived from the split alone.
// Each extent of what it reads, and each length of a view of what it reads
// that the host chose, is a candidate, a guess of its own; a candidate of one
// is one. ggml-symbolize-shapes derives the split from the guesses in its
// order and learns as it goes (`ggml.infer_symbols`): where an operation
// requires two extents equal (flash attention: the keys as many as the mask
// is wide; a rotation: a position per token; a reshape: its elements), two
// guesses are one, or a guess the number the model's side leaves it. Each
// class left that an input carries is a symbol, bounded by the room its views
// have, and the pass derives every other extent from them. The tensors are
// numbered as describeSplit numbers them.
//
// What `directory` gets: model.bc (the code, LLVM bitcode with hip-ep's
// artifact ABI) and constants.bin, model.ggml.mlir (the module it was
// compiled from), bindings.json (which described tensor each of the code's
// inputs and outputs is, and which outputs may be written in place),
// shapes.json (SplitShapes: each described tensor's sizes as expressions of
// the symbols), split.json (the description), symbols.json.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_FRONTEND_GGML_SPLITCOMPILER_H
#define HIP_FRONTEND_GGML_SPLITCOMPILER_H

#include "hip/Frontend/Ggml/CapturedGraph.h"
#include "hip/Frontend/Ggml/SplitGraph.h"

#include <memory>
#include <string>

namespace hip::ggml::capture {

struct SplitCapture {
  CapturedGraph graph;
  SplitDescription description;
  /// The symbols it is compiled with.
  SymbolSpec symbols;
  /// Why none were derived, when they were asked for and none were.
  std::string symbolsError;
};

/// Throws when the split cannot be captured (an operation the capture has no
/// semantics for).
std::unique_ptr<SplitCapture> captureSplit(const SplitNodes &split,
                                           const SplitTensors &tensors,
                                           const SymbolSpec *symbols);

/// Throws when it cannot be compiled.
void compileSplit(const SplitCapture &capture, const std::string &directory);

} // namespace hip::ggml::capture

#endif // HIP_FRONTEND_GGML_SPLITCOMPILER_H
