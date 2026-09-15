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

// Parse log level from string
SchemeLogLevel parseLogLevel(const std::string& level);

// Set global log level
void setSchemeLogLevel(SchemeLogLevel level);

// C type for Scheme FFI
typedef void* SchemeValue;

/// Encapsulates Chez Scheme runtime lifecycle.
/// Owned by HipsrDialect - one instance per dialect instance.
class ChezSchemeInterpreter {
public:
  explicit ChezSchemeInterpreter(SchemeLogLevel logLevel = SchemeLogLevel::Warning);
  ~ChezSchemeInterpreter();

  // Non-copyable, non-movable
  ChezSchemeInterpreter(const ChezSchemeInterpreter&) = delete;
  ChezSchemeInterpreter& operator=(const ChezSchemeInterpreter&) = delete;

  /// R5RS load: Load and evaluate a Scheme script file
  bool load(const char* scriptPath);

  /// R5RS eval: Evaluate Scheme code string
  bool eval(const char* code);

  /// Check if runtime is initialized
  bool isInitialized() const { return initialized; }

  /// Get current log level
  SchemeLogLevel getLogLevel() const { return logLevel; }

  // Create Scheme values from C++ primitives
  static SchemeValue makeString(const char* str);
  static SchemeValue makeInteger(long value);

  // Call a Scheme function with primitive arguments
  std::string callFunction(const char* functionName,
                          const std::vector<SchemeValue>& args);

  // Call a Scheme function with a single MLIR operation argument
  void callPassFunction(const char* functionName, mlir::Operation* op);

private:
  bool initialized = false;
  SchemeLogLevel logLevel;
};

} // namespace hipsr
} // namespace mlir

#endif
