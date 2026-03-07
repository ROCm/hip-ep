// Test: two chained hip.miopen.mul (pre-bufferized memref format)
//
//   mul0: tmp = A * B
//   mul1: D   = tmp * C
//
// Shapes: A, B, C, D are [1, 4, 4].  Pool: 64 bytes (1 temp buffer).

module {
  func.func @two_muls(
      %ctx: !hip.context,
      %A: memref<1x4x4xf32, strided<[?, ?, ?], offset: ?>>,
      %B: memref<1x4x4xf32, strided<[?, ?, ?], offset: ?>>,
      %C: memref<1x4x4xf32, strided<[?, ?, ?], offset: ?>>,
      %D: memref<1x4x4xf32>) {
    %pool = hip.alloc(%ctx) : memref<64xi8>
    %c0 = arith.constant 0 : index
    %tmp = memref.view %pool[%c0][] : memref<64xi8> to memref<1x4x4xf32>

    hip.miopen.mul(%ctx) ins(%A, %B : memref<1x4x4xf32, strided<[?, ?, ?], offset: ?>>, memref<1x4x4xf32, strided<[?, ?, ?], offset: ?>>) outs(%tmp : memref<1x4x4xf32>)
    hip.miopen.mul(%ctx) ins(%tmp, %C : memref<1x4x4xf32>, memref<1x4x4xf32, strided<[?, ?, ?], offset: ?>>) outs(%D : memref<1x4x4xf32>)

    hip.free(%ctx, %pool) : memref<64xi8>
    return
  }
}
