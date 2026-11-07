// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-fuse-rocmlir sinks the quantizes and rebiases the unsigned trees itself.
// Running hip-rocmlir-prepare-quant first must not change its result: a second
// sink finds nothing left to move, and a second rebias finds no unsigned tree.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %S/hip-fuse-rocmlir-sink-quantize.mlir > %t.sink.one
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant --hip-fuse-rocmlir %S/hip-fuse-rocmlir-sink-quantize.mlir > %t.sink.two
// RUN: diff %t.sink.one %t.sink.two

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %S/hip-fuse-rocmlir-rebias-unsigned.mlir > %t.rebias.one
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant --hip-fuse-rocmlir %S/hip-fuse-rocmlir-rebias-unsigned.mlir > %t.rebias.two
// RUN: diff %t.rebias.one %t.rebias.two

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %S/hip-fuse-rocmlir-qdq-conv.mlir > %t.qdq.one
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant --hip-fuse-rocmlir %S/hip-fuse-rocmlir-qdq-conv.mlir > %t.qdq.two
// RUN: diff %t.qdq.one %t.qdq.two

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %S/hip-fuse-rocmlir-qlinear-conv.mlir > %t.qconv.one
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant --hip-fuse-rocmlir %S/hip-fuse-rocmlir-qlinear-conv.mlir > %t.qconv.two
// RUN: diff %t.qconv.one %t.qconv.two
