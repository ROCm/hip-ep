// Test: two chained hip.hipblaslt.matmul (pre-bufferized memref format)
//
//   matmul0: tmp = A @ B0   [1,4,8] x [8,4] -> [1,4,4]
//   matmul1: C   = tmp @ B1 [1,4,4] x [4,4] -> [1,4,4]
//
// B0/B1 are 2D weights (broadcast across batch).
// Pool: 64 bytes (1 temp buffer [1,4,4]).

module {
  func.func @two_matmuls(
      %ctx: !hip.context,
      %A:  memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>,
      %B0: memref<8x4xf32, strided<[?, ?], offset: ?>>,
      %B1: memref<4x4xf32, strided<[?, ?], offset: ?>>,
      %C:  memref<1x4x4xf32>) {
    %pool = hip.alloc(%ctx) : memref<64xi8>
    %c0 = arith.constant 0 : index
    %tmp = memref.view %pool[%c0][] : memref<64xi8> to memref<1x4x4xf32>

    hip.hipblaslt.matmul(%ctx) ins(%A, %B0 : memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>, memref<8x4xf32, strided<[?, ?], offset: ?>>) outs(%tmp : memref<1x4x4xf32>)
    hip.hipblaslt.matmul(%ctx) ins(%tmp, %B1 : memref<1x4x4xf32>, memref<4x4xf32, strided<[?, ?], offset: ?>>) outs(%C : memref<1x4x4xf32>)

    hip.free(%ctx, %pool) : memref<64xi8>
    return
  }
}
