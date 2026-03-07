// Test: two chained hip.miopen.softmax (pre-bufferized memref format)
//
//   softmax0: tmp = softmax(A)
//   softmax1: B   = softmax(tmp)
//
// Shapes: A, B are [1, 4, 8].  Pool: 128 bytes (1 temp buffer).

module {
  func.func @two_softmaxes(
      %ctx: !hip.context,
      %A: memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>,
      %B: memref<1x4x8xf32>) {
    %pool = hip.alloc(%ctx) : memref<128xi8>
    %c0 = arith.constant 0 : index
    %tmp = memref.view %pool[%c0][] : memref<128xi8> to memref<1x4x8xf32>

    hip.miopen.softmax(%ctx) ins(%A : memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>) outs(%tmp : memref<1x4x8xf32>)
    hip.miopen.softmax(%ctx) ins(%tmp : memref<1x4x8xf32>) outs(%B : memref<1x4x8xf32>)

    hip.free(%ctx, %pool) : memref<128xi8>
    return
  }
}
