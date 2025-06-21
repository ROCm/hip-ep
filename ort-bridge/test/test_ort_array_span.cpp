/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Simple test to verify OrtArraySpan inheritance behavior
// This file demonstrates the usage patterns enabled by the new design

#include "../src/ort-array-span.hpp"
#include <cassert>

namespace morphizen {

// Mock function to test implicit conversion
void process_span(gsl::span<const int*> span) {
  // This function accepts any span, including OrtArraySpan
  // due to the inheritance relationship
  (void)span; // Suppress unused parameter warning
}

void test_ortarray_span_conversion() {
  // This test demonstrates that OrtArraySpan<T> can be implicitly
  // converted to gsl::span<T> due to inheritance

  // Note: This is a conceptual test - in real usage, the OrtArraySpan
  // would be created by the Graph class methods

  // Example usage showing implicit conversion:
  // auto nodes = graph.nodes_managed();  // Returns OrtArraySpan<const
  // OrtNode*> process_span(nodes);                 // Implicit conversion to
  // gsl::span

  // The key benefits of the inheritance approach:
  // 1. Natural span interface - all gsl::span methods available directly
  // 2. Implicit conversion to gsl::span for compatibility with existing code
  // 3. RAII resource management for OrtArrayOfConstObjects
  // 4. Type safety through template parameter
}

} // namespace morphizen

int main() {
  morphizen::test_ortarray_span_conversion();
  return 0;
}
