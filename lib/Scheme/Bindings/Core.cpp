/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


// Forward declarations from Logging.cpp
#define DEBUG_TYPE "scheme-core-bindings"

// Note: scheme.h included via SchemeMlirBindings.h -> ChezSchemeInterpreter.h

extern "C" {

const char* mlir_operation_get_name(uint64_t op) {
  if (!op) return "";
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return cppOp->getName().getStringRef().data();
}

// Get MLIRContext from operation
uint64_t mlir_operation_get_context(uint64_t op) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return reinterpret_cast<uint64_t>(cppOp->getContext());
}

// Get number of operands
int64_t mlir_operation_num_operands(uint64_t op) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return cppOp->getNumOperands();
}

// Get number of results
int64_t mlir_operation_num_results(uint64_t op) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return cppOp->getNumResults();
}

// Get operand at index
uint64_t mlir_operation_get_operand(uint64_t op, int64_t index) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  if (index < 0 || index >= (int64_t)cppOp->getNumOperands()) return 0;
  mlir::Value val = cppOp->getOperand(index);
  MlirValue cVal = wrap(val);
  return reinterpret_cast<uint64_t>(const_cast<void*>(cVal.ptr));
}

// Get result at index
uint64_t mlir_operation_get_result(uint64_t op, int64_t index) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  if (index < 0 || index >= (int64_t)cppOp->getNumResults()) return 0;
  mlir::Value val = cppOp->getResult(index);
  MlirValue cVal = wrap(val);
  return reinterpret_cast<uint64_t>(const_cast<void*>(cVal.ptr));
}

// Get the defining operation of a value (returns 0 for block arguments)
uint64_t mlir_value_get_defining_op(uint64_t value) {
  if (!value) return 0;
  MlirValue cVal{reinterpret_cast<const void*>(value)};
  mlir::Value val = unwrap(cVal);
  mlir::Operation* defOp = val.getDefiningOp();
  return reinterpret_cast<uint64_t>(defOp);
}

// Returns 1 if value is a block argument, 0 if it is an op result
int mlir_value_is_block_argument(uint64_t value) {
  if (!value) return 0;
  mlir::Value val = unwrap(MlirValue{reinterpret_cast<const void*>(value)});
  return mlir::isa<mlir::BlockArgument>(val) ? 1 : 0;
}

// Returns the result index of an OpResult value (-1 for block arguments)
int mlir_value_get_result_number(uint64_t value) {
  if (!value) return -1;
  mlir::Value val = unwrap(MlirValue{reinterpret_cast<const void*>(value)});
  auto result = mlir::dyn_cast<mlir::OpResult>(val);
  if (!result) return -1;
  return static_cast<int>(result.getResultNumber());
}

// Returns the number of DPS init (destination/outs) operands of an operation
int mlir_operation_num_dps_inits(uint64_t op_ptr) {
  if (!op_ptr) return 0;
  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op);
  if (!dpsOp) return 0;
  return static_cast<int>(dpsOp.getNumDpsInits());
}

// Returns the Value* of the i-th DPS init (outs) operand (0 if out of range)
uint64_t mlir_operation_get_dps_init_value(uint64_t op_ptr, int index) {
  if (!op_ptr) return 0;
  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op);
  if (!dpsOp) return 0;
  if (index < 0 || index >= static_cast<int>(dpsOp.getNumDpsInits())) return 0;
  mlir::Value v = dpsOp.getDpsInits()[index];
  return reinterpret_cast<uint64_t>(v.getAsOpaquePointer());
}

// Set the i-th operand of an operation to a new value
void mlir_operation_set_operand(uint64_t op_ptr, int index, uint64_t value) {
  if (!op_ptr || !value) return;
  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  mlir::Value val = unwrap(MlirValue{reinterpret_cast<const void*>(value)});
  op->setOperand(static_cast<unsigned>(index), val);
}

// Returns 1 if all results of the operation have no uses, 0 otherwise
int mlir_operation_use_empty(uint64_t op_ptr) {
  if (!op_ptr) return 1;
  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  return op->use_empty() ? 1 : 0;
}

// Walk operation tree and call Scheme callback for each operation
// callback: Scheme procedure (lambda (op) ...)
void mlir_operation_walk(uint64_t op, ptr callback) {
  if (!op) return;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);

  cppOp->walk([callback](mlir::Operation* walkOp) {
    ptr schemeOp = Sunsigned64(reinterpret_cast<uint64_t>(walkOp));
    Scall1(callback, schemeOp);
  });
}


// Logging functions callable from Scheme
ptr mlir_operation_get_parent(ptr op_ptr) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  mlir::Operation* parent = op->getParentOp();
  return parent;
}

ptr mlir_operation_get_operand_value(ptr op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  if (index < 0 || index >= (int)op->getNumOperands())
    return nullptr;
  mlir::Value operand = op->getOperand(index);
  return const_cast<void*>(operand.getAsOpaquePointer());
}

ptr mlir_operation_get_result_value(ptr op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  if (index < 0 || index >= (int)op->getNumResults())
    return nullptr;
  mlir::Value result = op->getResult(index);
  return const_cast<void*>(result.getAsOpaquePointer());
}

ptr mlir_operation_get_loc(ptr op_ptr) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  return const_cast<void*>(op->getLoc().getAsOpaquePointer());
}

ptr mlir_operation_get_block_argument(ptr op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  // Walk up to parent function
  while (op && !llvm::isa<mlir::func::FuncOp>(op)) {
    op = op->getParentOp();
  }
  if (!op)
    return nullptr;

  auto funcOp = llvm::cast<mlir::func::FuncOp>(op);
  if (index < 0 || index >= (int)funcOp.getNumArguments())
    return nullptr;

  mlir::Value arg = funcOp.getArgument(index);
  return const_cast<void*>(arg.getAsOpaquePointer());
}

//===----------------------------------------------------------------------===//
// Phase 3 & 4: Explicit builder API
//===----------------------------------------------------------------------===//

// Create an operation at the current rewriter insertion point.
// Caller must set the insertion point explicitly before calling.
// For hipsr.placeholder: automatically adds the shape region and placeholder_type attr.
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
    uint64_t v = Sunsigned64_value(Scar(cur));
    operands.push_back(mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(v)));
  }
  for (ptr cur = static_cast<ptr>(result_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op: bad result types list"); return 0; }
    uint64_t t = Sunsigned64_value(Scar(cur));
    resultTypes.push_back(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(t)));
  }

  rewriter->setInsertionPoint(loc_op);

  mlir::OperationState state(loc_op->getLoc(), op_name);
  state.addOperands(operands);
  state.addTypes(resultTypes);

  // hipsr.placeholder requires an empty shape region and placeholder_type attr
  if (std::string_view(op_name) == "hipsr.placeholder") {
    state.addRegion();
    state.addAttribute("placeholder_type",
        mlir::hipsr::PlaceholderTypeAttr::get(loc_op->getContext(),
                                               mlir::hipsr::PlaceholderType::Normal));
  }

  return reinterpret_cast<uint64_t>(rewriter->create(state));
}

// Like mlir_build_op but pre-allocates num_regions empty regions in the OperationState.
// Required for ops that verify they have exactly N regions at creation time
// (e.g. shape.assuming, scf.if) when those regions are populated afterwards via :regions.
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
    uint64_t v = Sunsigned64_value(Scar(cur));
    operands.push_back(mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(v)));
  }
  for (ptr cur = static_cast<ptr>(result_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op_with_regions: bad result types list"); return 0; }
    uint64_t t = Sunsigned64_value(Scar(cur));
    resultTypes.push_back(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(t)));
  }

  rewriter->setInsertionPoint(loc_op);

  mlir::OperationState state(loc_op->getLoc(), op_name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  for (int i = 0; i < num_regions; ++i)
    state.addRegion();

  // hipsr.placeholder also needs its placeholder_type attribute
  if (std::string_view(op_name) == "hipsr.placeholder") {
    state.addAttribute("placeholder_type",
        mlir::hipsr::PlaceholderTypeAttr::get(loc_op->getContext(),
                                               mlir::hipsr::PlaceholderType::Normal));
  }

  return reinterpret_cast<uint64_t>(rewriter->create(state));
}

// Like mlir_build_op but takes a plain OpBuilder* (e.g. from mlir_builder_at_block_end).
// Used for ops inside region blocks where a fresh OpBuilder is used instead of the rewriter.
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
    uint64_t v = Sunsigned64_value(Scar(cur));
    operands.push_back(mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(v)));
  }
  for (ptr cur = static_cast<ptr>(result_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op_in_block: bad result types"); return 0; }
    uint64_t t = Sunsigned64_value(Scar(cur));
    resultTypes.push_back(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(t)));
  }

  mlir::OperationState state(loc_op->getLoc(), op_name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  return reinterpret_cast<uint64_t>(builder->create(state));
}

// Like mlir_build_op_in_block but pre-allocates num_regions empty regions.
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
    uint64_t v = Sunsigned64_value(Scar(cur));
    operands.push_back(mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(v)));
  }
  for (ptr cur = static_cast<ptr>(result_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) { mlir_log_error("mlir_build_op_in_block_with_regions: bad result types"); return 0; }
    uint64_t t = Sunsigned64_value(Scar(cur));
    resultTypes.push_back(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(t)));
  }

  mlir::OperationState state(loc_op->getLoc(), op_name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  for (int i = 0; i < num_regions; ++i)
    state.addRegion();

  return reinterpret_cast<uint64_t>(builder->create(state));
}

// Set rewriter insertion point to immediately before op.
void mlir_set_insertion_point_before(uint64_t rewriter_ptr, uint64_t op_ptr) {
  if (!rewriter_ptr || !op_ptr) return;
  reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr)
      ->setInsertionPoint(reinterpret_cast<mlir::Operation*>(op_ptr));
}

// Set rewriter insertion point to the end of a block.
void mlir_set_insertion_point_to_block_end(uint64_t rewriter_ptr, uint64_t block_ptr) {
  if (!rewriter_ptr || !block_ptr) return;
  reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr)
      ->setInsertionPointToEnd(reinterpret_cast<mlir::Block*>(block_ptr));
}

// Get the i-th region of an operation.
uint64_t mlir_op_get_region(uint64_t op_ptr, int region_idx) {
  if (!op_ptr) return 0;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  if (region_idx < 0 || region_idx >= (int)op->getNumRegions()) return 0;
  return reinterpret_cast<uint64_t>(&op->getRegion(region_idx));
}

// Create a new block in a region with the given argument types.
// Sets the rewriter insertion point to the end of the new block.
// arg_types_list: Scheme list of type uptrs (stored as Sunsigned64).
uint64_t mlir_region_create_block(uint64_t rewriter_ptr, uint64_t region_ptr,
                                   ptr arg_types_list) {
  if (!rewriter_ptr || !region_ptr) return 0;
  // Must use RewriterBase::createBlock (not OpBuilder::createBlock) so the
  // conversion framework receives proper block-creation notifications.
  auto* rewriter = reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr);
  auto* region   = reinterpret_cast<mlir::Region*>(region_ptr);
  mlir::Location loc = region->getParentOp()->getLoc();

  mlir::Block* block = rewriter->createBlock(region);
  for (ptr cur = static_cast<ptr>(arg_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) break;
    uint64_t t = Sunsigned64_value(Scar(cur));
    block->addArgument(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(t)), loc);
  }
  rewriter->setInsertionPointToEnd(block);
  return reinterpret_cast<uint64_t>(block);
}

// Get the i-th argument of a block as a Value opaque pointer.
uint64_t mlir_block_get_argument(uint64_t block_ptr, int idx) {
  if (!block_ptr) return 0;
  auto* block = reinterpret_cast<mlir::Block*>(block_ptr);
  if (idx < 0 || idx >= (int)block->getNumArguments()) return 0;
  return reinterpret_cast<uint64_t>(block->getArgument(idx).getAsOpaquePointer());
}

// Get the shape::ShapeType from an MLIRContext.
// replaceOp/eraseOp are on RewriterBase, not OpBuilder — these must use RewriterBase*.
int mlir_replace_op(uint64_t rewriter_ptr, uint64_t old_op_ptr, uint64_t new_value_ptr) {
  if (!rewriter_ptr) { mlir_log_error("mlir_replace_op: no rewriter"); return 0; }
  auto* rewriter = reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr);
  auto* op  = reinterpret_cast<mlir::Operation*>(old_op_ptr);
  auto  val = mlir::Value::getFromOpaquePointer(reinterpret_cast<void*>(new_value_ptr));
  rewriter->replaceOp(op, val);
  return 1;
}

int mlir_erase_op(uint64_t rewriter_ptr, uint64_t op_ptr) {
  if (!rewriter_ptr) { mlir_log_error("mlir_erase_op: no rewriter"); return 0; }
  reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr)
      ->eraseOp(reinterpret_cast<mlir::Operation*>(op_ptr));
  return 1;
}

void mlir_notify_match_failure(uint64_t op_ptr, const char* reason) {
  mlir_log_debug((std::string("Pattern match failure: ") + reason).c_str());
}

// Direct erase without a rewriter — for post-pass cleanup outside a pattern callback.
void mlir_op_erase(uint64_t op_ptr) {
  if (!op_ptr) return;
  reinterpret_cast<mlir::Operation*>(op_ptr)->erase();
}

//===----------------------------------------------------------------------===//
// Pattern Registration - Scheme-defined patterns
//===----------------------------------------------------------------------===//


void mlir_operation_set_attr(uint64_t op, const char* attr_name, int64_t value) {
  if (!op) return;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  mlir::MLIRContext* ctx = cppOp->getContext();
  mlir::IntegerAttr attr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), value);
  cppOp->setAttr(attr_name, attr);
}

void mlir_operation_set_index_attr(uint64_t op_ptr, const char* attr_name, int64_t value) {
  if (!op_ptr) return;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op_ptr);
  mlir::IntegerAttr attr = mlir::IntegerAttr::get(
      mlir::IndexType::get(cppOp->getContext()), value);
  cppOp->setAttr(attr_name, attr);
}

// Read a single integer attribute; returns default_val if absent.
int64_t mlir_operation_get_integer_attr(uint64_t op_ptr, const char* attr_name, int64_t default_val) {
  if (!op_ptr) return default_val;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(attr_name))
    return attr.getInt();
  return default_val;
}

// Read a dense-i64 or array-of-integer-attr as a Scheme list. Returns Snil when absent.
ptr mlir_operation_get_integer_array_attr(uint64_t op_ptr, const char* attr_name) {
  if (!op_ptr) return Snil;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  if (auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(attr_name)) {
    ptr list = Snil;
    for (int i = (int)attr.size() - 1; i >= 0; --i)
      list = Scons(Sinteger(attr[i]), list);
    return list;
  }
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>(attr_name)) {
    ptr list = Snil;
    for (int i = (int)attr.size() - 1; i >= 0; --i) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr[i]);
      if (!intAttr) return Snil;
      list = Scons(Sinteger(intAttr.getInt()), list);
    }
    return list;
  }
  return Snil;
}

// Set a DenseI64ArrayAttr on an operation. values_list is a Scheme list of fixnums.
void mlir_operation_set_dense_i64_array(uint64_t op_ptr, const char* attr_name, ptr values_list) {
  if (!op_ptr) return;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  llvm::SmallVector<int64_t> values;
  for (ptr cur = static_cast<ptr>(values_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) break;
    values.push_back(Sinteger_value(Scar(cur)));
  }
  op->setAttr(attr_name, mlir::DenseI64ArrayAttr::get(op->getContext(), values));
}

// Change a hipsr.placeholder's placeholder_type attribute to Barrier.
void mlir_operation_copy_attr(uint64_t dst_op_ptr, const char* dst_name,
                               uint64_t src_op_ptr, const char* src_name) {
  if (!dst_op_ptr || !src_op_ptr) return;
  auto* dst = reinterpret_cast<mlir::Operation*>(dst_op_ptr);
  auto* src = reinterpret_cast<mlir::Operation*>(src_op_ptr);
  auto attr = src->getAttr(src_name);
  if (attr) dst->setAttr(dst_name, attr);
}

// Returns 1 if type is a RankedTensorType with device memory space, 0 otherwise.
int mlir_operation_has_attr(uint64_t op_ptr, const char* attr_name) {
  if (!op_ptr) return 0;
  return reinterpret_cast<mlir::Operation*>(op_ptr)->hasAttr(attr_name) ? 1 : 0;
}


// Create a block in a region with given arg types.
// Does NOT change any rewriter's insertion point.
uint64_t mlir_new_block(uint64_t region_ptr, ptr arg_types_list) {
  if (!region_ptr) return 0;
  auto* region = reinterpret_cast<mlir::Region*>(region_ptr);
  auto* block = new mlir::Block();
  region->push_back(block);
  mlir::Location loc = region->getParentOp()->getLoc();
  for (ptr cur = static_cast<ptr>(arg_types_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) break;
    uint64_t t = Sunsigned64_value(Scar(cur));
    block->addArgument(mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(t)), loc);
  }
  return reinterpret_cast<uint64_t>(block);
}

// Create a heap-allocated OpBuilder positioned at the end of a block.
// Independent of any ConversionPatternRewriter — does not affect its insertion point.
// Caller must destroy with mlir_destroy_builder.
uint64_t mlir_builder_at_block_end(uint64_t block_ptr) {
  if (!block_ptr) return 0;
  auto* block = reinterpret_cast<mlir::Block*>(block_ptr);
  auto* builder = new mlir::OpBuilder(block, block->end());
  return reinterpret_cast<uint64_t>(builder);
}

// Destroy a builder created by mlir_builder_at_block_end.
void mlir_destroy_builder(uint64_t builder_ptr) {
  if (!builder_ptr) return;
  delete reinterpret_cast<mlir::OpBuilder*>(builder_ptr);
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerCoreBindings() {
  Sregister_symbol("mlir_operation_get_name", (void*)::mlir_operation_get_name);
  Sregister_symbol("mlir_operation_get_context", (void*)::mlir_operation_get_context);
  Sregister_symbol("mlir_operation_num_operands", (void*)::mlir_operation_num_operands);
  Sregister_symbol("mlir_operation_num_results", (void*)::mlir_operation_num_results);
  Sregister_symbol("mlir_operation_get_operand", (void*)::mlir_operation_get_operand);
  Sregister_symbol("mlir_operation_get_result", (void*)::mlir_operation_get_result);
  Sregister_symbol("mlir_value_get_defining_op", (void*)::mlir_value_get_defining_op);
  Sregister_symbol("mlir_value_is_block_argument", (void*)::mlir_value_is_block_argument);
  Sregister_symbol("mlir_value_get_result_number", (void*)::mlir_value_get_result_number);
  Sregister_symbol("mlir_operation_num_dps_inits", (void*)::mlir_operation_num_dps_inits);
  Sregister_symbol("mlir_operation_get_dps_init_value", (void*)::mlir_operation_get_dps_init_value);
  Sregister_symbol("mlir_operation_set_operand", (void*)::mlir_operation_set_operand);
  Sregister_symbol("mlir_operation_use_empty", (void*)::mlir_operation_use_empty);
  Sregister_symbol("mlir_operation_walk", (void*)::mlir_operation_walk);
  Sregister_symbol("mlir_operation_get_parent", (void*)::mlir_operation_get_parent);
  Sregister_symbol("mlir_operation_get_operand_value", (void*)::mlir_operation_get_operand_value);
  Sregister_symbol("mlir_operation_get_result_value", (void*)::mlir_operation_get_result_value);
  Sregister_symbol("mlir_operation_get_loc", (void*)::mlir_operation_get_loc);
  Sregister_symbol("mlir_operation_get_block_argument", (void*)::mlir_operation_get_block_argument);
  Sregister_symbol("mlir_build_op", (void*)::mlir_build_op);
  Sregister_symbol("mlir_build_op_with_regions", (void*)::mlir_build_op_with_regions);
  Sregister_symbol("mlir_set_insertion_point_before", (void*)::mlir_set_insertion_point_before);
  Sregister_symbol("mlir_set_insertion_point_to_block_end", (void*)::mlir_set_insertion_point_to_block_end);
  Sregister_symbol("mlir_op_get_region", (void*)::mlir_op_get_region);
  Sregister_symbol("mlir_region_create_block", (void*)::mlir_region_create_block);
  Sregister_symbol("mlir_block_get_argument", (void*)::mlir_block_get_argument);
  Sregister_symbol("mlir_new_block", (void*)::mlir_new_block);
  Sregister_symbol("mlir_builder_at_block_end", (void*)::mlir_builder_at_block_end);
  Sregister_symbol("mlir_destroy_builder", (void*)::mlir_destroy_builder);
  Sregister_symbol("mlir_build_op_in_block", (void*)::mlir_build_op_in_block);
  Sregister_symbol("mlir_build_op_in_block_with_regions", (void*)::mlir_build_op_in_block_with_regions);
  Sregister_symbol("mlir_replace_op", (void*)::mlir_replace_op);
  Sregister_symbol("mlir_erase_op", (void*)::mlir_erase_op);
  Sregister_symbol("mlir_notify_match_failure", (void*)::mlir_notify_match_failure);
  Sregister_symbol("mlir_op_erase", (void*)::mlir_op_erase);
  Sregister_symbol("mlir_operation_set_attr",       (void*)::mlir_operation_set_attr);
  Sregister_symbol("mlir_operation_set_index_attr", (void*)::mlir_operation_set_index_attr);
  Sregister_symbol("mlir_operation_get_integer_attr", (void*)::mlir_operation_get_integer_attr);
  Sregister_symbol("mlir_operation_get_integer_array_attr", (void*)::mlir_operation_get_integer_array_attr);
  Sregister_symbol("mlir_operation_set_dense_i64_array", (void*)::mlir_operation_set_dense_i64_array);
  Sregister_symbol("mlir_operation_copy_attr", (void*)::mlir_operation_copy_attr);
  Sregister_symbol("mlir_operation_has_attr", (void*)::mlir_operation_has_attr);
}

} // namespace hipsr
} // namespace mlir
