/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::hipsr;

#include "hip/Dialect/Hipsr/IR/HipsrDialect.cpp.inc"

// Enum code first: the attribute's parser/printer below calls these
// enum name<->value helpers.
#include "hip/Dialect/Hipsr/IR/HipsrEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrTypes.cpp.inc"

void HipsrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hip/Dialect/Hipsr/IR/HipsrAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "hip/Dialect/Hipsr/IR/HipsrTypes.cpp.inc"
      >();
}

llvm::MemoryBuffer *HipsrDialect::getOrLoadFileMap(llvm::StringRef path) {
  // The dialect is a shared singleton and MLIR runs passes multi-threaded;
  // guard the cache against concurrent lookups/inserts.
  std::lock_guard<std::mutex> lock(fileMapsMutex);

  auto it = fileMaps.find(path);
  if (it != fileMaps.end()) {
    return it->second.get();
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufOr =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false);
  if (!bufOr) {
    return nullptr;
  }

  llvm::MemoryBuffer *raw = bufOr->get();
  fileMaps[path] = std::move(*bufOr);
  return raw;
}

namespace {

void populateHipsrToLLVMPatterns(const LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  populateHipsrConstantLoweringPatterns(typeConverter, patterns);
  populateHipsrAddLoweringPatterns(typeConverter, patterns);
  populateHipsrGetPoolLoweringPatterns(typeConverter, patterns);
  populateHipsrCastLoweringPatterns(typeConverter, patterns);
}

struct HipsrConvertToLLVMInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    // #hipsr.mem<kind> -> integer address space. MemorySpaceKind's numeric
    // values already match the AMDGPU address spaces (host=0, device=1,
    // pinned=2, managed=3), so map the enum directly.
    typeConverter.addTypeAttributeConversion(
        [](BaseMemRefType,
           MemorySpaceAttr space) -> TypeConverter::AttributeConversionResult {
          return IntegerAttr::get(IntegerType::get(space.getContext(), 64),
                                  static_cast<int64_t>(space.getKind()));
        });
    typeConverter.addConversion([](ContextType type) -> Type {
      return LLVM::LLVMPointerType::get(type.getContext(), 0);
    });
    target.addIllegalDialect<HipsrDialect>();
    populateHipsrToLLVMPatterns(typeConverter, patterns);
  }
};

} // namespace

void mlir::hipsr::registerConvertHipsrToLLVMInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *, HipsrDialect *dialect) {
    dialect->addInterfaces<HipsrConvertToLLVMInterface>();
  });
}
