/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_LOCKED_SCHEME_OBJECT_H
#define LIB_DIALECT_HIPSR_SCHEME_LOCKED_SCHEME_OBJECT_H

// Include Chez Scheme types via wrapper
#include "hip/Dialect/Hipsr/Scheme/Runtime/SchemeWrapper.h"

namespace mlir {
namespace hipsr {

/// @brief RAII wrapper for Scheme object GC locking
///
/// Locks a Scheme object on construction, unlocks on destruction.
/// Prevents the garbage collector from moving or collecting the object.
///
/// Use this when storing Scheme objects (especially callbacks) in C++ data
/// structures that outlive a single FFI call. The lock prevents the GC from:
/// 1. Moving the object in memory (invalidating C pointers to it)
/// 2. Collecting the object (causing use-after-free)
///
/// Example usage:
/// @code
///   class SchemeConversionPattern {
///     LockedSchemeObject callback_;  // Automatically locked/unlocked
///   public:
///     SchemeConversionPattern(ptr cb) : callback_(cb) {}
///     // Destructor automatically unlocks callback
///   };
/// @endcode
///
/// @note Non-copyable, non-movable (like std::lock_guard)
/// @note Null/false pointers are safely handled (no-op)
class LockedSchemeObject {
 public:
  /// @brief Lock a Scheme object
  /// @param obj Scheme object (ptr) to lock, can be null/Sfalse
  explicit LockedSchemeObject(ptr obj) : obj_(obj) {
    if (obj_ && obj_ != Sfalse) {
      Slock_object(obj_);
    }
  }

  /// @brief Unlock the Scheme object
  ~LockedSchemeObject() {
    if (obj_ && obj_ != Sfalse) {
      Sunlock_object(obj_);
    }
  }

  // Non-copyable
  LockedSchemeObject(const LockedSchemeObject&) = delete;
  LockedSchemeObject& operator=(const LockedSchemeObject&) = delete;

  // Non-movable (moving would require updating lock count)
  LockedSchemeObject(LockedSchemeObject&&) = delete;
  LockedSchemeObject& operator=(LockedSchemeObject&&) = delete;

  /// @brief Get the wrapped Scheme object
  /// @return The locked ptr
  ptr get() const { return obj_; }

  /// @brief Implicit conversion to ptr for convenience
  /// @return The locked ptr
  operator ptr() const { return obj_; }

 private:
  ptr obj_;
};

}  // namespace hipsr
}  // namespace mlir

#endif
