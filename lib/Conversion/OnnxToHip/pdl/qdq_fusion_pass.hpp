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

// Native rewrite callback - marks operation with attribute
// CRITICAL: Check if already marked to prevent infinite loop!
inline void markQDQFusionRewrite(mlir::PatternRewriter& rewriter,
                                  mlir::Operation* rootOp,
                                  mlir::Attribute nameAttr)
{
    // Check if already marked - prevent infinite loop
    if (rootOp->hasAttr("hip_fusion")) {
        return; // Already marked, skip
    }
    
    auto strAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    if (!strAttr)
        return;
    
    rewriter.modifyOpInPlace(rootOp, [&] {
        rootOp->setAttr("hip_fusion", strAttr);
    });
}

// Apply PDL patterns
inline bool run(mlir::ModuleOp mlirModule, llvm::StringRef pdlBytecodeFile)
{
    if (pdlBytecodeFile.empty())
        return true;

    mlir::MLIRContext* ctx = mlirModule.getContext();
    ctx->loadDialect<mlir::pdl::PDLDialect>();
    ctx->loadDialect<mlir::pdl_interp::PDLInterpDialect>();

    mlir::ParserConfig parseConfig(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> pdlModule =
        mlir::parseSourceFile<mlir::ModuleOp>(pdlBytecodeFile, parseConfig);
    if (!pdlModule)
        return false;

    mlir::PDLPatternModule pdlPatterns(std::move(pdlModule));
    pdlPatterns.registerRewriteFunction("markQDQFusion", markQDQFusionRewrite);

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
