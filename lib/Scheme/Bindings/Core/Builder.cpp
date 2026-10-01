/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Mirrors (mlir core builder): builder API, block/region primitives,
// rewriter ops, and type constructors.

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

extern "C" {

// Create a generic MLIR operation via a RewriterBase, setting the insertion
// point immediately before loc_op and borrowing its location.
// rewriter_ptr:       RewriterBase* as uptr (ConversionPatternRewriter or similar)
// loc_op_ptr:         Operation* used as insertion point and location source
// op_name:            registered MLIR op name (e.g. "arith.constant")
// operands_list:      Scheme list of Value* uptrs (each Sunsigned64)
// result_types_list:  Scheme list of Type* uptrs (each Sunsigned64)
// Returns: Operation* as uptr, or 0 on bad input.
// Note: hipsr.placeholder automatically receives one empty region and a
//       placeholder_type = Normal attribute.
uint64_t mlir_build_op(uint64_t rewriter_ptr, uint64_t loc_op_ptr,
                        const char* op_name,
                        ptr operands_list, ptr result_types_list) {
  if (!rewriter_ptr || !loc_op_ptr) return 0;
  auto* rewriter = reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr);
  auto* loc_op   = reinterpret_cast<mlir::Operation*>(loc_op_ptr);

  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<mlir::Type>  resultTypes;

  for (ptr cur = static_cast<ptr>(operands_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op: bad operands list"); return 0; }
    operands.push_back(mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(Sunsigned64_value(Scar(cur)))));
  }
  for (ptr cur = static_cast<ptr>(result_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op: bad result types list"); return 0; }
    resultTypes.push_back(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(Sunsigned64_value(Scar(cur)))));
  }

  rewriter->setInsertionPoint(loc_op);
  mlir::OperationState state(loc_op->getLoc(), op_name);
  state.addOperands(operands);
  state.addTypes(resultTypes);

  if (std::string_view(op_name) == "hipsr.placeholder") {
    state.addRegion();
    state.addAttribute("placeholder_type",
        mlir::hipsr::PlaceholderTypeAttr::get(loc_op->getContext(),
                                               mlir::hipsr::PlaceholderType::Normal));
  }
  return reinterpret_cast<uint64_t>(rewriter->create(state));
}

// Like mlir_build_op but pre-allocates num_regions empty regions in the
// OperationState before creating the op. Required for ops that verify they
// have exactly N regions at creation time (e.g. scf.if, scf.for).
// rewriter_ptr:       RewriterBase* as uptr
// loc_op_ptr:         Operation* for insertion point and location
// op_name:            registered MLIR op name
// operands_list:      Scheme list of Value* uptrs
// result_types_list:  Scheme list of Type* uptrs
// num_regions:        number of empty regions to add
// Returns: Operation* as uptr, or 0 on bad input.
uint64_t mlir_build_op_with_regions(uint64_t rewriter_ptr, uint64_t loc_op_ptr,
                                    const char* op_name,
                                    ptr operands_list, ptr result_types_list,
                                    int num_regions) {
  if (!rewriter_ptr || !loc_op_ptr) return 0;
  auto* rewriter = reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr);
  auto* loc_op   = reinterpret_cast<mlir::Operation*>(loc_op_ptr);

  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<mlir::Type>  resultTypes;

  for (ptr cur = static_cast<ptr>(operands_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op_with_regions: bad operands list"); return 0; }
    operands.push_back(mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(Sunsigned64_value(Scar(cur)))));
  }
  for (ptr cur = static_cast<ptr>(result_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op_with_regions: bad result types list"); return 0; }
    resultTypes.push_back(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(Sunsigned64_value(Scar(cur)))));
  }

  rewriter->setInsertionPoint(loc_op);
  mlir::OperationState state(loc_op->getLoc(), op_name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  for (int i = 0; i < num_regions; ++i)
    state.addRegion();

  if (std::string_view(op_name) == "hipsr.placeholder") {
    state.addAttribute("placeholder_type",
        mlir::hipsr::PlaceholderTypeAttr::get(loc_op->getContext(),
                                               mlir::hipsr::PlaceholderType::Normal));
  }
  return reinterpret_cast<uint64_t>(rewriter->create(state));
}

// Create a generic MLIR operation via a plain OpBuilder (not a RewriterBase).
// Used inside region blocks where a block-level builder is active rather than
// a conversion rewriter. The location is taken from loc_op.
// builder_ptr:        OpBuilder* as uptr (from mlir_builder_at_block_end)
// loc_op_ptr:         Operation* supplying the source location
// op_name:            registered MLIR op name
// operands_list:      Scheme list of Value* uptrs
// result_types_list:  Scheme list of Type* uptrs
// Returns: Operation* as uptr, or 0 on bad input.
uint64_t mlir_build_op_in_block(uint64_t builder_ptr, uint64_t loc_op_ptr,
                                  const char* op_name,
                                  ptr operands_list, ptr result_types_list) {
  if (!builder_ptr || !loc_op_ptr) return 0;
  auto* builder = reinterpret_cast<mlir::OpBuilder*>(builder_ptr);
  auto* loc_op  = reinterpret_cast<mlir::Operation*>(loc_op_ptr);

  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<mlir::Type>  resultTypes;

  for (ptr cur = static_cast<ptr>(operands_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op_in_block: bad operands"); return 0; }
    operands.push_back(mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(Sunsigned64_value(Scar(cur)))));
  }
  for (ptr cur = static_cast<ptr>(result_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op_in_block: bad result types"); return 0; }
    resultTypes.push_back(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(Sunsigned64_value(Scar(cur)))));
  }

  mlir::OperationState state(loc_op->getLoc(), op_name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  return reinterpret_cast<uint64_t>(builder->create(state));
}

// Like mlir_build_op_in_block but also pre-allocates num_regions empty regions.
// builder_ptr:        OpBuilder* as uptr
// loc_op_ptr:         Operation* supplying the source location
// op_name:            registered MLIR op name
// operands_list:      Scheme list of Value* uptrs
// result_types_list:  Scheme list of Type* uptrs
// num_regions:        number of empty regions to add
// Returns: Operation* as uptr, or 0 on bad input.
uint64_t mlir_build_op_in_block_with_regions(uint64_t builder_ptr, uint64_t loc_op_ptr,
                                               const char* op_name,
                                               ptr operands_list, ptr result_types_list,
                                               int num_regions) {
  if (!builder_ptr || !loc_op_ptr) return 0;
  auto* builder = reinterpret_cast<mlir::OpBuilder*>(builder_ptr);
  auto* loc_op  = reinterpret_cast<mlir::Operation*>(loc_op_ptr);

  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<mlir::Type>  resultTypes;

  for (ptr cur = static_cast<ptr>(operands_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op_in_block_with_regions: bad operands"); return 0; }
    operands.push_back(mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(Sunsigned64_value(Scar(cur)))));
  }
  for (ptr cur = static_cast<ptr>(result_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op_in_block_with_regions: bad result types"); return 0; }
    resultTypes.push_back(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(Sunsigned64_value(Scar(cur)))));
  }

  mlir::OperationState state(loc_op->getLoc(), op_name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  for (int i = 0; i < num_regions; ++i)
    state.addRegion();
  return reinterpret_cast<uint64_t>(builder->create(state));
}

// Create a generic MLIR operation via a plain OpBuilder with optional regions.
// Lower-level than mlir_build_op: does not set the insertion point and does
// not go through a RewriterBase. Used by Scheme's mlir-create-op wrapper when
// an explicit builder and location op are provided.
// builder_ptr:        OpBuilder* as uptr
// loc_op_ptr:         Operation* supplying the source location
// op_name:            registered MLIR op name
// operands_list:      Scheme list of Value* uptrs
// result_types_list:  Scheme list of Type* uptrs
// num_regions:        number of empty regions to pre-allocate (0 for none)
// Returns: Operation* as uptr, or 0 on bad input.
uint64_t mlir_create_op(uint64_t builder_ptr, uint64_t loc_op_ptr,
                         const char* op_name,
                         ptr operands_list, ptr result_types_list,
                         int num_regions) {
  if (!builder_ptr || !loc_op_ptr) return 0;
  auto* builder = reinterpret_cast<mlir::OpBuilder*>(builder_ptr);
  auto* loc_op  = reinterpret_cast<mlir::Operation*>(loc_op_ptr);

  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<mlir::Type>  resultTypes;

  for (ptr cur = static_cast<ptr>(operands_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_create_op: bad operands list"); return 0; }
    operands.push_back(mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(Sunsigned64_value(Scar(cur)))));
  }
  for (ptr cur = static_cast<ptr>(result_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_create_op: bad result types list"); return 0; }
    resultTypes.push_back(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(Sunsigned64_value(Scar(cur)))));
  }

  mlir::OperationState state(loc_op->getLoc(), op_name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  for (int i = 0; i < num_regions; ++i)
    state.addRegion();

  if (std::string_view(op_name) == "hipsr.placeholder") {
    state.addAttribute("placeholder_type",
        mlir::hipsr::PlaceholderTypeAttr::get(loc_op->getContext(),
                                               mlir::hipsr::PlaceholderType::Normal));
  }
  return reinterpret_cast<uint64_t>(builder->create(state));
}

// Move the rewriter insertion point to immediately before op.
// rewriter_ptr:  RewriterBase* as uptr
// op_ptr:        Operation* to insert before
void mlir_set_insertion_point_before(uint64_t rewriter_ptr, uint64_t op_ptr) {
  if (!rewriter_ptr || !op_ptr) return;
  reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr)
      ->setInsertionPoint(reinterpret_cast<mlir::Operation*>(op_ptr));
}

// Move the rewriter insertion point to the end of a block.
// rewriter_ptr:  RewriterBase* as uptr
// block_ptr:     Block* as uptr
void mlir_set_insertion_point_to_block_end(uint64_t rewriter_ptr, uint64_t block_ptr) {
  if (!rewriter_ptr || !block_ptr) return;
  reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr)
      ->setInsertionPointToEnd(reinterpret_cast<mlir::Block*>(block_ptr));
}

// Get the i-th region of an operation.
// op_ptr:      Operation* as uptr
// region_idx:  0-based region index
// Returns: Region* as uptr, or 0 if op is null or index out of range.
uint64_t mlir_op_get_region(uint64_t op_ptr, int region_idx) {
  if (!op_ptr) return 0;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  if (region_idx < 0 || region_idx >= (int)op->getNumRegions()) return 0;
  return reinterpret_cast<uint64_t>(&op->getRegion(region_idx));
}

// Create a new block inside a region using the rewriter, add typed arguments,
// and set the rewriter insertion point to the end of the new block.
// rewriter_ptr:    RewriterBase* as uptr
// region_ptr:      Region* as uptr
// arg_types_list:  Scheme list of Type* uptrs for the block's arguments
// Returns: Block* as uptr, or 0 on null input.
uint64_t mlir_region_create_block(uint64_t rewriter_ptr, uint64_t region_ptr,
                                   ptr arg_types_list) {
  if (!rewriter_ptr || !region_ptr) return 0;
  auto* rewriter = reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr);
  auto* region   = reinterpret_cast<mlir::Region*>(region_ptr);
  mlir::Location loc = region->getParentOp()->getLoc();

  mlir::Block* block = rewriter->createBlock(region);
  for (ptr cur = static_cast<ptr>(arg_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) break;
    block->addArgument(mlir::Type::getFromOpaquePointer(
        reinterpret_cast<const void*>(Sunsigned64_value(Scar(cur)))), loc);
  }
  rewriter->setInsertionPointToEnd(block);
  return reinterpret_cast<uint64_t>(block);
}

// Get the idx-th argument of a block as a Value* opaque pointer.
// block_ptr:  Block* as uptr
// idx:        0-based argument index
// Returns: Value opaque ptr, or 0 if block is null or index out of range.
uint64_t mlir_block_get_argument(uint64_t block_ptr, int idx) {
  if (!block_ptr) return 0;
  auto* block = reinterpret_cast<mlir::Block*>(block_ptr);
  if (idx < 0 || idx >= (int)block->getNumArguments()) return 0;
  return reinterpret_cast<uint64_t>(block->getArgument(idx).getAsOpaquePointer());
}

// Replace all uses of old_op's results with new_value, then erase old_op.
// rewriter_ptr:    RewriterBase* as uptr (must be non-null)
// old_op_ptr:      Operation* to replace
// new_value_ptr:   Value opaque ptr to substitute for old_op's single result
// Returns: 1 on success, 0 if no rewriter.
int mlir_replace_op(uint64_t rewriter_ptr, uint64_t old_op_ptr, uint64_t new_value_ptr) {
  if (!rewriter_ptr) { mlir_log_error("mlir_replace_op: no rewriter"); return 0; }
  reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr)
      ->replaceOp(reinterpret_cast<mlir::Operation*>(old_op_ptr),
                  mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(new_value_ptr)));
  return 1;
}

// Erase an operation through a rewriter (notifies listeners).
// rewriter_ptr:  RewriterBase* as uptr (must be non-null)
// op_ptr:        Operation* to erase
// Returns: 1 on success, 0 if no rewriter.
int mlir_erase_op(uint64_t rewriter_ptr, uint64_t op_ptr) {
  if (!rewriter_ptr) { mlir_log_error("mlir_erase_op: no rewriter"); return 0; }
  reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr)
      ->eraseOp(reinterpret_cast<mlir::Operation*>(op_ptr));
  return 1;
}

// Create a new bare Block, append it to a region, add typed arguments, and
// return it. Does NOT use a rewriter — for use with with-block-builder where
// a plain OpBuilder manages insertion.
// region_ptr:      Region* as uptr
// arg_types_list:  Scheme list of Type* uptrs for block arguments
// Returns: Block* as uptr, or 0 if region is null.
uint64_t mlir_new_block(uint64_t region_ptr, ptr arg_types_list) {
  if (!region_ptr) return 0;
  auto* region = reinterpret_cast<mlir::Region*>(region_ptr);
  auto* block = new mlir::Block();
  region->push_back(block);
  mlir::Location loc = region->getParentOp()->getLoc();
  for (ptr cur = static_cast<ptr>(arg_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) break;
    block->addArgument(mlir::Type::getFromOpaquePointer(
        reinterpret_cast<const void*>(Sunsigned64_value(Scar(cur)))), loc);
  }
  return reinterpret_cast<uint64_t>(block);
}

// Heap-allocate an OpBuilder positioned at the end of a block.
// Must be destroyed with mlir_destroy_builder when done.
// block_ptr:  Block* as uptr
// Returns: OpBuilder* as uptr, or 0 if block is null.
uint64_t mlir_builder_at_block_end(uint64_t block_ptr) {
  if (!block_ptr) return 0;
  auto* block = reinterpret_cast<mlir::Block*>(block_ptr);
  return reinterpret_cast<uint64_t>(new mlir::OpBuilder(block, block->end()));
}

// Destroy an OpBuilder created by mlir_builder_at_block_end.
// builder_ptr:  OpBuilder* as uptr (no-op if 0)
void mlir_destroy_builder(uint64_t builder_ptr) {
  if (!builder_ptr) return;
  delete reinterpret_cast<mlir::OpBuilder*>(builder_ptr);
}

// Get the MLIRContext* from any MLIR Type*.
// type_ptr:  Type opaque ptr (Type::getAsOpaquePointer())
// Returns: MLIRContext* as uptr, or 0 if type_ptr is 0.
uint64_t mlir_type_get_context(uint64_t type_ptr) {
  if (!type_ptr) return 0;
  return reinterpret_cast<uint64_t>(
      mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(type_ptr))
          .getContext());
}

// Get the MLIR index type for a context.
// ctx_ptr:  MLIRContext* as uptr
// Returns: Type opaque ptr for mlir::IndexType, or 0 if ctx is null.
uint64_t mlir_get_index_type(uint64_t ctx_ptr) {
  if (!ctx_ptr) return 0;
  return reinterpret_cast<uint64_t>(
      mlir::IndexType::get(reinterpret_cast<mlir::MLIRContext*>(ctx_ptr)).getAsOpaquePointer());
}

// Get the MLIR i64 integer type for a context.
// ctx_ptr:  MLIRContext* as uptr
// Returns: Type opaque ptr for mlir::IntegerType<64>, or 0 if ctx is null.
uint64_t mlir_get_i64_type(uint64_t ctx_ptr) {
  if (!ctx_ptr) return 0;
  return reinterpret_cast<uint64_t>(
      mlir::IntegerType::get(reinterpret_cast<mlir::MLIRContext*>(ctx_ptr), 64).getAsOpaquePointer());
}

// Get the MLIR i1 integer type (boolean) for a context.
// ctx_ptr:  MLIRContext* as uptr
// Returns: Type opaque ptr for mlir::IntegerType<1>, or 0 if ctx is null.
uint64_t mlir_get_i1_type(uint64_t ctx_ptr) {
  if (!ctx_ptr) return 0;
  return reinterpret_cast<uint64_t>(
      mlir::IntegerType::get(reinterpret_cast<mlir::MLIRContext*>(ctx_ptr), 1).getAsOpaquePointer());
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerBuilderBindings() {
  Sregister_symbol("mlir_build_op",                          (void*)::mlir_build_op);
  Sregister_symbol("mlir_build_op_with_regions",             (void*)::mlir_build_op_with_regions);
  Sregister_symbol("mlir_build_op_in_block",                 (void*)::mlir_build_op_in_block);
  Sregister_symbol("mlir_build_op_in_block_with_regions",    (void*)::mlir_build_op_in_block_with_regions);
  Sregister_symbol("mlir_create_op",                         (void*)::mlir_create_op);
  Sregister_symbol("mlir_set_insertion_point_before",        (void*)::mlir_set_insertion_point_before);
  Sregister_symbol("mlir_set_insertion_point_to_block_end",  (void*)::mlir_set_insertion_point_to_block_end);
  Sregister_symbol("mlir_op_get_region",                     (void*)::mlir_op_get_region);
  Sregister_symbol("mlir_region_create_block",               (void*)::mlir_region_create_block);
  Sregister_symbol("mlir_block_get_argument",                (void*)::mlir_block_get_argument);
  Sregister_symbol("mlir_replace_op",                        (void*)::mlir_replace_op);
  Sregister_symbol("mlir_erase_op",                          (void*)::mlir_erase_op);
  Sregister_symbol("mlir_new_block",                         (void*)::mlir_new_block);
  Sregister_symbol("mlir_builder_at_block_end",              (void*)::mlir_builder_at_block_end);
  Sregister_symbol("mlir_destroy_builder",                   (void*)::mlir_destroy_builder);
  Sregister_symbol("mlir_type_get_context",                   (void*)::mlir_type_get_context);
  Sregister_symbol("mlir_get_index_type",                    (void*)::mlir_get_index_type);
  Sregister_symbol("mlir_get_i64_type",                      (void*)::mlir_get_i64_type);
  Sregister_symbol("mlir_get_i1_type",                       (void*)::mlir_get_i1_type);
}

} // namespace hipsr
} // namespace mlir
