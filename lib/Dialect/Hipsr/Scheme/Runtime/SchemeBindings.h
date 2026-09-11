/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_SCHEMERUNTIME_H
#define LIB_DIALECT_HIPSR_SCHEME_SCHEMERUNTIME_H

#include <string>
#include <vector>
#include <functional>

// C type for Scheme FFI - must be at global scope for extern "C" functions
typedef void* SchemeValue;

namespace mlir {
class Operation;
class Value;
class Type;
class Attribute;

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

// Initialize Scheme runtime and register MLIR FFI bindings
bool initializeSchemeRuntime(SchemeLogLevel logLevel = SchemeLogLevel::Warning);

// Call a Scheme function with primitive arguments (legacy API)
std::string callSchemeFunction(const char* functionName,
                                const std::vector<SchemeValue>& args);

// Create Scheme values from C++ primitives
SchemeValue makeSchemeString(const char* str);
SchemeValue makeSchemeInteger(long value);

// MLIR C++ to Scheme conversions - wrap MLIR objects as foreign pointers
SchemeValue makeSchemeOperation(mlir::Operation* op);
SchemeValue makeSchemeValue(mlir::Value val);
SchemeValue makeSchemeType(mlir::Type type);
SchemeValue makeSchemeAttribute(mlir::Attribute attr);

// Register MLIR foreign functions accessible from Scheme
void registerMlirForeignFunctions();

// Load and evaluate a Scheme script file
bool loadSchemeScript(const char* scriptPath);

// Call a Scheme function with a single MLIR operation argument
// Used to invoke Scheme-defined pass entry points
void callSchemePassFunction(const char* functionName, mlir::Operation* op);

//===----------------------------------------------------------------------===//
// Phase 1: Type System FFI
//===----------------------------------------------------------------------===//

// Check if a Type is a RankedTensorType
// Returns: boolean (1 = true, 0 = false)
int mlir_type_is_ranked_tensor(SchemeValue type_ptr);

// Get element type of a tensor type
// Returns: Type* as uptr
SchemeValue mlir_type_get_element_type(SchemeValue type_ptr);

// Get shape of a ranked tensor type
// Returns: list of dimension sizes (int64_t*)
SchemeValue mlir_type_get_shape(SchemeValue type_ptr);

// Get rank of a ranked tensor type
// Returns: int
int mlir_type_get_rank(SchemeValue type_ptr);

// Get type from a Value
// Returns: Type* as uptr
SchemeValue mlir_value_get_type(SchemeValue value_ptr);

//===----------------------------------------------------------------------===//
// Phase 2: Operation/Value Navigation FFI
//===----------------------------------------------------------------------===//

// Get parent operation
// Returns: Operation* as uptr (or nullptr)
SchemeValue mlir_operation_get_parent(SchemeValue op_ptr);

// Get operand Value from operation by index
// Returns: Value as uptr
SchemeValue mlir_operation_get_operand_value(SchemeValue op_ptr, int index);

// Get result Value from operation by index
// Returns: Value as uptr
SchemeValue mlir_operation_get_result_value(SchemeValue op_ptr, int index);

// Get location from operation
// Returns: Location* as uptr
SchemeValue mlir_operation_get_loc(SchemeValue op_ptr);

// Get block argument from parent function
// Returns: Value as uptr
SchemeValue mlir_operation_get_block_argument(SchemeValue op_ptr, int index);

} // namespace hipsr
} // namespace mlir

//===----------------------------------------------------------------------===//
// Phase 3-4: IR Construction and Pattern Rewriter FFI (extern "C")
//===----------------------------------------------------------------------===//

extern "C" {

// Phase 3: IR Construction FFI (OpBuilder)
// Note: These functions require a PatternRewriter context. For now, they are
// placeholders that will be properly integrated when called from pattern passes.

// Create hipsr.placeholder operation
SchemeValue mlir_create_placeholder_op(SchemeValue ctx_value,
                                       SchemeValue input_value,
                                       SchemeValue result_type,
                                       int placeholder_type_int);

// Create hipsr.cast operation
SchemeValue mlir_create_cast_op(SchemeValue ctx_value,
                                 SchemeValue input_value,
                                 SchemeValue output_value,
                                 SchemeValue result_type);

// Phase 4: Pattern Rewriter FFI

// Replace operation with a value
int mlir_replace_op(SchemeValue old_op, SchemeValue new_value);

// Erase operation
int mlir_erase_op(SchemeValue op);

// Notify match failure
void mlir_notify_match_failure(SchemeValue op, const char* reason);

} // extern "C"

#endif
