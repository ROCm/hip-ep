#include "scheme.h"
#include <iostream>
#include <cstdint>

extern "C" {

// This function CRASHES when declared in Scheme
static void crash_func(uint64_t x) {
  std::cerr << "crash_func called with " << x << "\n";
}

// This function WORKS fine
static void working_func(uint64_t x) {
  std::cerr << "working_func called with " << x << "\n";
}

static void register_functions() {
  Sregister_symbol("crash_func", (void*)crash_func);
  Sregister_symbol("working_func", (void*)working_func);
}

} // extern "C"

// Entry point
extern "C" void setup() {
  register_functions();
  std::cerr << "Functions registered\n";
}
