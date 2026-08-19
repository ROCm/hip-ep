// qdq_fusion_pass.hpp — PDL fusion pass for QDQ MatMul patterns

#pragma once

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace hip {
namespace pdl {

// Native constraint: Get context argument from function
// Low-level signature required by MLIR PDL infrastructure
inline mlir::LogicalResult getContextArg(mlir::PatternRewriter& rewriter,
                                          mlir::PDLResultList& results,
                                          llvm::ArrayRef<mlir::PDLValue> args) {
    // args[0] should be the operation
    if (args.size() != 1)
        return mlir::failure();
    
    auto *op = args[0].dyn_cast<mlir::Operation*>();
    if (!op)
        return mlir::failure();
    
    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    if (!funcOp || funcOp.getNumArguments() == 0)
        return mlir::failure();
    
    // Return the context (first function argument) as a Value
    results.push_back(funcOp.getArgument(0));
    return mlir::success();
}

// Native constraint: Extract float value from onnx.Constant
// Low-level signature required by MLIR PDL infrastructure
inline mlir::LogicalResult extractScaleValue(mlir::PatternRewriter& rewriter,
                                               mlir::PDLResultList& results,
                                               llvm::ArrayRef<mlir::PDLValue> args) {
    // args[0] should be the constant value
    if (args.size() != 1)
        return mlir::failure();
    
    auto constValue = args[0].dyn_cast<mlir::Value>();
    if (!constValue)
        return mlir::failure();
    
    auto defOp = constValue.getDefiningOp();
    if (!defOp)
        return mlir::failure();
    
    auto valueAttr = defOp->getAttr("value");
    if (!valueAttr)
        return mlir::failure();
    
    auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr);
    if (!denseAttr || !denseAttr.isSplat())
        return mlir::failure();
    
    float scaleValue = denseAttr.getSplatValue<mlir::FloatAttr>().getValueAsDouble();
    
    // Return the extracted float as an f32 attribute
    results.push_back(rewriter.getF32FloatAttr(scaleValue));
    return mlir::success();
}

// Apply PDL patterns
inline bool run(mlir::ModuleOp mlirModule, llvm::StringRef pdlBytecodeFile)
{
    if (pdlBytecodeFile.empty())
        return true;

    mlir::MLIRContext* ctx = mlirModule.getContext();

    mlir::ParserConfig parseConfig(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> pdlModule =
        mlir::parseSourceFile<mlir::ModuleOp>(pdlBytecodeFile, parseConfig);
    if (!pdlModule)
        return false;

    mlir::PDLPatternModule pdlPatterns(std::move(pdlModule));
    
    // Register native constraints with low-level signatures
    pdlPatterns.registerConstraintFunction("GetContextArg", getContextArg);
    pdlPatterns.registerConstraintFunction("ExtractScaleValue", extractScaleValue);

    mlir::RewritePatternSet patterns(ctx);
    patterns.add(std::move(pdlPatterns));

    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    // Walk all FuncOps and apply patterns
    bool ok = true;
    mlirModule.walk([&](mlir::func::FuncOp funcOp) {
        if (mlir::failed(mlir::applyPatternsGreedily(funcOp, frozen)))
            ok = false;
    });
    return ok;
}

} // namespace pdl
} // namespace hip
