// Test: two chained hip.miopen.softmax (pre-bufferized memref format)
//
//   softmax0: tmp = softmax(A)
//   softmax1: B   = softmax(tmp)
//
// Shapes: A, B are [1, 4, 8].  Pool: 128 bytes (1 temp buffer).

module {
  func.func @two_softmaxes(
      %A: memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>,
      %B: memref<1x4x8xf32>) {
    %handle = hip.create_handle() : !hip.handle
    %pool = hip.alloc(%handle) : memref<128xi8>
    %c0 = arith.constant 0 : index
    %tmp = memref.view %pool[%c0][] : memref<128xi8> to memref<1x4x8xf32>

    hip.miopen.softmax(%handle) ins(%A : memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>) outs(%tmp : memref<1x4x8xf32>)
    hip.miopen.softmax(%handle) ins(%tmp : memref<1x4x8xf32>) outs(%B : memref<1x4x8xf32>)

    hip.free(%handle, %pool) : memref<128xi8>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
