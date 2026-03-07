// Test: two chained hip.miopen.rms_norm (pre-bufferized memref format)
//
//   norm0: tmp = RMSNorm(A, W0)
//   norm1: B   = RMSNorm(tmp, W1)
//
// Shapes: A, B are [1, 4, 8]; W0, W1 are [8].  Pool: 128 bytes (1 temp buffer).

module {
  func.func @two_rms_norms(
      %ctx: !hip.context,
      %A:  memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>,
      %W0: memref<8xf32, strided<[?], offset: ?>>,
      %W1: memref<8xf32, strided<[?], offset: ?>>,
      %B:  memref<1x4x8xf32>) {
    %pool = hip.alloc(%ctx) : memref<128xi8>
    %c0 = arith.constant 0 : index
    %tmp = memref.view %pool[%c0][] : memref<128xi8> to memref<1x4x8xf32>

    hip.miopen.rms_norm(%ctx) ins(%A, %W0 : memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>, memref<8xf32, strided<[?], offset: ?>>) outs(%tmp : memref<1x4x8xf32>)
    hip.miopen.rms_norm(%ctx) ins(%tmp, %W1 : memref<1x4x8xf32>, memref<8xf32, strided<[?], offset: ?>>) outs(%B : memref<1x4x8xf32>)

    hip.free(%ctx, %pool) : memref<128xi8>
    return
  }
}
