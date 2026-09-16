// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -hipsr-populate-shape-region | FileCheck %s

// The scatter only overwrites slices, so the region hands the data's shape
// straight back and every dynamic extent stays symbolic. The indices and
// updates never reach the placeholder, because neither changes the shape.
// CHECK-LABEL: func.func @scatter_features(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[EMBEDS:.+]]: tensor<?x?x4096xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %[[POSITIONS:.+]]: tensor<?x3xi64, #hipsr.mem<device>>,
// CHECK-SAME:    %[[FEATURES:.+]]: tensor<?xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[EMBEDS]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x?x4096xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[EMBEDS_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT:      hipsr.shape_yield %[[EMBEDS_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.+]] = hipsr.scatter_nd(%[[CTX]]) ins(%[[EMBEDS]], %[[POSITIONS]], %[[FEATURES]] : tensor<?x?x4096xf16, #hipsr.mem<device>>, tensor<?x3xi64, #hipsr.mem<device>>, tensor<?xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x?x4096xf16, #hipsr.mem<device>>) : tensor<?x?x4096xf16, #hipsr.mem<device>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scatter_features(%ctx: !hipsr.context,
                            %embeds: tensor<?x?x4096xf16, #hipsr.mem<device>>,
                            %positions: tensor<?x3xi64, #hipsr.mem<device>>,
                            %features: tensor<?xf16, #hipsr.mem<device>>) {
  %init = hipsr.placeholder(%ctx)
      ins(%embeds : tensor<?x?x4096xf16, #hipsr.mem<device>>)
      {placeholder_type = #hipsr.placeholder_type<normal>}
      : tensor<?x?x4096xf16, #hipsr.mem<device>>
  %result = hipsr.scatter_nd(%ctx)
      ins(%embeds, %positions, %features
          : tensor<?x?x4096xf16, #hipsr.mem<device>>,
            tensor<?x3xi64, #hipsr.mem<device>>,
            tensor<?xf16, #hipsr.mem<device>>)
      outs(%init : tensor<?x?x4096xf16, #hipsr.mem<device>>)
      : tensor<?x?x4096xf16, #hipsr.mem<device>>
  return
}
