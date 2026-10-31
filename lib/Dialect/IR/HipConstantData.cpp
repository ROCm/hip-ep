/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipConstantData.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <memory>

using namespace mlir;
using namespace mlir::hip;

FailureOr<DenseElementsAttr>
mlir::hip::readConstantData(ConstantOp constant, RankedTensorType type) {
  switch (constant.getSourceKind()) {
  case ConstantOp::SourceKind::Inline: {
    // The verifier ties an inline value to the carrier's result type, but a
    // failed match has to be cheaper than an assertion if it ever does not.
    auto value = dyn_cast<DenseElementsAttr>(constant.getValueAttr());
    if (!value || value.getType() != type)
      return failure();
    return value;
  }
  case ConstantOp::SourceKind::Memory:
    return failure();
  case ConstantOp::SourceKind::File:
    break;
  }

  const int64_t size = constant.getSizeAttr().getInt();
  const int64_t elementBytes = (type.getElementTypeBitWidth() + 7) / 8;
  if (!type.hasStaticShape() || size != type.getNumElements() * elementBytes)
    return failure();

  StringRef path = constant.getLocationAttr().getValue();
  const int64_t offset = constant.getOffsetAttr().getInt();
  // The range is validated against the file before mapping: a slice running
  // past the end can otherwise be mapped and fault on access rather than
  // failing here.
  uint64_t fileSize = 0;
  if (offset < 0 || llvm::sys::fs::file_size(path, fileSize) ||
      static_cast<uint64_t>(offset) + static_cast<uint64_t>(size) > fileSize)
    return failure();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFileSlice(path, static_cast<uint64_t>(size),
                                       offset);
  if (!buffer || (*buffer)->getBufferSize() != static_cast<uint64_t>(size))
    return failure();
  ArrayRef<char> raw((*buffer)->getBufferStart(), static_cast<size_t>(size));
  if (!DenseElementsAttr::isValidRawBuffer(type, raw))
    return failure();
  DenseElementsAttr value = DenseElementsAttr::getFromRawBuffer(type, raw);
  if (!value)
    return failure();
  return value;
}
