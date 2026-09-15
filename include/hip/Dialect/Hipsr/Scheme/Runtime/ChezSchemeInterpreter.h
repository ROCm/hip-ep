/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_CHEZSCHEMEINTERPRETER_H
#define LIB_DIALECT_HIPSR_SCHEME_CHEZSCHEMEINTERPRETER_H

#include <string>
#include <vector>

namespace mlir {
class Operation;

namespace hipsr {

// Log levels for Scheme logging
enum class SchemeLogLevel {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warning = 3,
  Error = 4,
  Fatal = 5
};

// C type for Scheme FFI
typedef void* SchemeValue;

/// Singleton Chez Scheme runtime.
/// All methods are static. Runtime is initialized once globally.
class ChezSchemeInterpreter {
 public:
  // Initialize the Scheme runtime (called once)
  static void initialize(SchemeLogLevel logLevel = SchemeLogLevel::Warning);

  /// Check if runtime is initialized
  static bool isInitialized() { return initialized; }

  /// Shutdown the Scheme runtime
  static void shutdown();

  // Parse log level from string
  static SchemeLogLevel parseLogLevel(const std::string& level);

  // Set global log level
  static void setLogLevel(SchemeLogLevel level);

  // Get current log level
  static SchemeLogLevel getLogLevel() { return logLevel; }

  /// R5RS load: Load and evaluate a Scheme script file
  static bool load(const char* scriptPath);

  /// R5RS eval: Evaluate Scheme code string
  static bool eval(const char* code);

  // Create Scheme values from C++ primitives
  static SchemeValue makeString(const char* str);
  static SchemeValue makeInteger(long value);

  // Call a Scheme function with primitive arguments
  static std::string callFunction(const char* functionName,
                                  const std::vector<SchemeValue>& args);

  // Call a Scheme function with a single MLIR operation argument
  static void callPassFunction(const char* functionName, mlir::Operation* op);

 private:
  // Singleton - deleted constructors
  ChezSchemeInterpreter() = delete;
  ~ChezSchemeInterpreter() = delete;
  ChezSchemeInterpreter(const ChezSchemeInterpreter&) = delete;
  ChezSchemeInterpreter& operator=(const ChezSchemeInterpreter&) = delete;

  static bool initialized;
  static SchemeLogLevel logLevel;
};

}  // namespace hipsr
}  // namespace mlir

#endif
