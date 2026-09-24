module {
  func.func @main_graph(
      %ctx: !hip.context,
      %lhs: memref<1x4x4xf32, strided<[?, ?, ?], offset: ?>>,
      %rhs: memref<1x4x4xf32, strided<[?, ?, ?], offset: ?>>) -> memref<1x4x4xf32> {
    %output = hip.alloc_output(%ctx) {out_idx = 0 : i64} : memref<1x4x4xf32>
    hip.add(%ctx)
        ins(%lhs, %rhs : memref<1x4x4xf32, strided<[?, ?, ?], offset: ?>>,
            memref<1x4x4xf32, strided<[?, ?, ?], offset: ?>>)
        outs(%output : memref<1x4x4xf32>)
    return %output : memref<1x4x4xf32>
  }
}
