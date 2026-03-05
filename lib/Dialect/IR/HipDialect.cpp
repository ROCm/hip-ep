/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hip;

#include "hip/Dialect/IR/HipDialect.cpp.inc"

void HipDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hip/Dialect/IR/HipTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/IR/HipOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// MemoryEffectsOpInterface Implementations
//===----------------------------------------------------------------------===//

void mlir::hip::GetPoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // Returns a view of pre-allocated GPU memory pool (read-only from IR
  // perspective; actual GPU memory is managed by the runtime)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::ConvOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // Read inputs (conservative: assumes reads from memory)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  // Write output
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::GemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // Read inputs and write output (result is read-write due to beta != 0)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::MaxPoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::AvgPoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::ReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::MatMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::GroupQueryAttentionOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::MulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::SubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::GatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::ReduceSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::SigmoidOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::RotaryEmbeddingOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::SimplifiedLayerNormOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::SkipSimplifiedLayerNormOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void mlir::hip::CastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// DestinationStyleOpInterface Implementations
//===----------------------------------------------------------------------===//

MutableOperandRange mlir::hip::ConvOp::getDpsInitsMutable() {
  return getOutputMutable(); // Last operand is destination
}

MutableOperandRange mlir::hip::ReluOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::GemmOp::getDpsInitsMutable() {
  return getResultMutable();
}

MutableOperandRange mlir::hip::MaxPoolOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::AvgPoolOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::MatMulOp::getDpsInitsMutable() {
  return getResultMutable();
}

MutableOperandRange mlir::hip::GroupQueryAttentionOp::getDpsInitsMutable() {
  // All 3 destination operands: output, present_key, present_value (contiguous)
  return MutableOperandRange(*this, 8, 3);
}

MutableOperandRange mlir::hip::MulOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::SubOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::GatherOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::ReduceSumOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::SigmoidOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::CastOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::RotaryEmbeddingOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::SimplifiedLayerNormOp::getDpsInitsMutable() {
  return getOutputMutable();
}

MutableOperandRange mlir::hip::SkipSimplifiedLayerNormOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, 4, 2);
}

void mlir::hip::GetConstantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // Allocates view to constant memory (read-only effect)
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

// Type and op class implementations (parse/print/verify, TypeIDs)
#define GET_TYPEDEF_CLASSES
#include "hip/Dialect/IR/HipTypes.cpp.inc"

#define GET_OP_CLASSES
#include "hip/Dialect/IR/HipOps.cpp.inc"
