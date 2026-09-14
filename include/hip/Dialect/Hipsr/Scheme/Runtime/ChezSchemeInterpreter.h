/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_CHEZSCHEMEINTERPRETER_H
#define LIB_DIALECT_HIPSR_SCHEME_CHEZSCHEMEINTERPRETER_H

#include <string>

namespace mlir {
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

/// Encapsulates Chez Scheme runtime lifecycle.
/// Owned by HipsrDialect - one instance per dialect instance.
/// Handles initialization, script loading, and cleanup.
class ChezSchemeInterpreter {
public:
  /// Initialize Scheme runtime with specified log level.
  /// Returns true on success, false on failure.
  explicit ChezSchemeInterpreter(SchemeLogLevel logLevel = SchemeLogLevel::Warning);

  /// Cleanup: shutdown Scheme runtime
  ~ChezSchemeInterpreter();

  // Non-copyable, non-movable
  ChezSchemeInterpreter(const ChezSchemeInterpreter&) = delete;
  ChezSchemeInterpreter& operator=(const ChezSchemeInterpreter&) = delete;

  /// Load and evaluate a Scheme script file
  /// Returns true on success, false on failure
  bool loadScript(const char* scriptPath);

  /// Evaluate Scheme code string
  /// Returns true on success, false on failure
  bool evaluateCode(const char* code);

  /// Check if runtime is initialized
  bool isInitialized() const { return initialized; }

  /// Get current log level
  SchemeLogLevel getLogLevel() const { return logLevel; }

private:
  bool initialized = false;
  SchemeLogLevel logLevel;
};

} // namespace hipsr
} // namespace mlir

#endif // LIB_DIALECT_HIPSR_SCHEME_CHEZSCHEMEINTERPRETER_H
