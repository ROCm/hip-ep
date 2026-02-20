//===- main_e2e.cpp - Main driver for E2E transformer test ----------------===//
//
// Calls the MLIR-compiled @run function from test_e2e.mlir.
// The function allocates all GPU buffers internally and exercises every
// HIP dialect op in a complete transformer layer loop.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

extern "C" __declspec(dllimport) void run();

int main() {
  printf("=== HIP Dialect E2E Transformer Test ===\n");
  printf("Calling MLIR-compiled @run()...\n\n");

  run();

  printf("\n=== All ops executed successfully. ===\n");
  return 0;
}
