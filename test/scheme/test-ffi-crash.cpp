// Minimal test to isolate FFI crash
#include "scheme.h"
#include <iostream>

// Test 1: Simplest void function
extern "C" void test_void_simple(uint64_t x) {
  std::cerr << "test_void_simple called with " << x << "\n";
}

// Test 2: Void function that matches working pattern
extern "C" void test_void_like_log(const char* msg) {
  std::cerr << "test_void_like_log: " << msg << "\n";
}

// Test 3: No-op function
extern "C" void test_void_noop(uint64_t x) {
  // Do absolutely nothing
}

// Test 4: Function that declares but doesn't use MLIR types
struct DummyMLIRType;
extern "C" void test_void_with_mlir_headers(uint64_t x) {
  DummyMLIRType* ptr = reinterpret_cast<DummyMLIRType*>(x);
  (void)ptr; // Suppress warning
  std::cerr << "test_void_with_mlir_headers called\n";
}

// Test 5: Function that casts but doesn't use the pointer
extern "C" void test_void_with_cast(uint64_t x) {
  auto ptr = reinterpret_cast<void*>(x);
  std::cerr << "test_void_with_cast: ptr=" << ptr << "\n";
}

// Register all test functions
extern "C" void register_test_functions() {
  Sregister_symbol("test_void_simple", (void*)test_void_simple);
  Sregister_symbol("test_void_like_log", (void*)test_void_like_log);
  Sregister_symbol("test_void_noop", (void*)test_void_noop);
  Sregister_symbol("test_void_with_mlir_headers", (void*)test_void_with_mlir_headers);
  Sregister_symbol("test_void_with_cast", (void*)test_void_with_cast);
}
