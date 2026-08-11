// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: mkdir -p %t/out
// RUN: hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/out' %t/valid.mlir | FileCheck %s --check-prefix=EXTERNAL
// RUN: %python %S/../Inputs/check_hip_constant_artifact.py %t/out/model.constants.bin %t/out/model.constants.json
// RUN: hip-mlir-opt --hip-externalize-constants %t/default-inline.mlir | FileCheck %s --check-prefix=INLINE
// RUN: not hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/out' %t/stale.mlir 2>&1 | FileCheck %s --check-prefix=STALE
// RUN: not hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/out' %t/collision.mlir 2>&1 | FileCheck %s --check-prefix=COLLISION
// RUN: not hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/out' %t/missing-file.mlir 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/out skip-constant-data=true' %t/missing-file.mlir | FileCheck %s --check-prefix=STREAM
// RUN: not hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/out' %t/memory-source.mlir 2>&1 | FileCheck %s --check-prefix=MEMORY
// RUN: printf '\376\377\000\200\000\200\377\377' > %t/typed.bin
// RUN: cd %t && hip-mlir-opt --hip-externalize-constants typed-inline.mlir | FileCheck %s --check-prefix=TYPED

// This file contains only HIP/builtin dialect operations. It models constants
// emitted by an AfterConvertOnnxToHip plugin without any ONNX dependency.

// EXTERNAL: module attributes
// EXTERNAL-SAME: hip.constants_file = "model.constants.bin"
// EXTERNAL-SAME: hipdnn.constant_offsets = array<i64: 0, 64>
// EXTERNAL-SAME: hipdnn.constant_sizes = array<i64: 1, 4>
// EXTERNAL-DAG: memref.global "private" @hip_ext_constant_plugin_weight_0 : memref<si8>
// EXTERNAL-DAG: memref.global "private" @hip_ext_constant_1 : memref<4xui8>
// EXTERNAL-NOT: hip.constant

// INLINE-NOT: hip.constants_file
// INLINE-NOT: memref.global
// INLINE: arith.constant dense<3> : tensor<si8>
// INLINE-NOT: hip.constant

// STALE: error: hip-externalize-constants found stale `hipdnn.constant_sizes` metadata
// COLLISION: error: externalized constant symbol collision: @hip_ext_constant_0
// MISSING: error: failed to open external data file:
// STREAM: hipdnn.constant_file_paths = ["/definitely/missing/hip-constant.bin"]
// STREAM-SAME: hipdnn.constant_source_kinds = array<i32: 2>
// MEMORY: error: memory-address sources require production externalization with an injected FileSystem
// TYPED-LABEL: func.func @typed_inline
// TYPED: arith.constant dense<[-2, -32768]> : tensor<2xsi16>
// TYPED: arith.constant dense<[32768, 65535]> : tensor<2xui16>
// TYPED: return {{.*}} : tensor<2xsi16>, tensor<2xui16>

//--- valid.mlir
module {
  func.func @plugin_constants() -> (tensor<si8>, tensor<4xui8>) {
    %0 = hip.constant {
      onnx_node_name = "/plugin/weight",
      value = dense<7> : tensor<si8>
    } : tensor<si8>
    %1 = hip.constant {
      value = dense<[1, 2, 3, 4]> : tensor<4xui8>
    } : tensor<4xui8>
    return %0, %1 : tensor<si8>, tensor<4xui8>
  }
}

//--- default-inline.mlir
module {
  func.func @default_inline() -> tensor<si8> {
    %0 = hip.constant {value = dense<3> : tensor<si8>} : tensor<si8>
    return %0 : tensor<si8>
  }
}

//--- stale.mlir
module attributes {hipdnn.constant_sizes = array<i64: 4>} {
  func.func @stale() -> tensor<4xi8> {
    %0 = hip.constant {value = dense<[1, 2, 3, 4]> : tensor<4xi8>} : tensor<4xi8>
    return %0 : tensor<4xi8>
  }
}

//--- collision.mlir
module {
  func.func private @hip_ext_constant_0()
  func.func @collision() -> tensor<4xi8> {
    %0 = hip.constant {value = dense<[1, 2, 3, 4]> : tensor<4xi8>} : tensor<4xi8>
    return %0 : tensor<4xi8>
  }
}

//--- missing-file.mlir
module {
  func.func @missing_file() -> tensor<4xi8> {
    %0 = hip.constant {
      location = "/definitely/missing/hip-constant.bin",
      offset = 0 : i64,
      size = 4 : i64
    } : tensor<4xi8>
    return %0 : tensor<4xi8>
  }
}

//--- typed-inline.mlir
module {
  func.func @typed_inline() -> (tensor<2xsi16>, tensor<2xui16>) {
    %s = hip.constant {
      location = "typed.bin", offset = 0 : i64, size = 4 : i64
    } : tensor<2xsi16>
    %u = hip.constant {
      location = "typed.bin", offset = 4 : i64, size = 4 : i64
    } : tensor<2xui16>
    return %s, %u : tensor<2xsi16>, tensor<2xui16>
  }
}

//--- memory-source.mlir
module {
  func.func @memory_source() -> tensor<4xi8> {
    %0 = hip.constant {
      location = "*/_ORT_MEM_ADDR_/*", offset = 1 : i64, size = 4 : i64
    } : tensor<4xi8>
    return %0 : tensor<4xi8>
  }
}
