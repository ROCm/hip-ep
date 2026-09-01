/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- GatherConversion.cpp - Convert onnx.Gather to hipsr ---------------===//
//
// onnx.Gather selects entries of `data` along `axis` at the positions in
// `indices`. Where the data lives picks the form:
//
// - Device data becomes hipsr.gather, a runtime call. Its placeholder gets no
//   shape region here: hipsr.gather is DPS, so hipsr-populate-shape-region
//   fills it in later, as for every DPS op.
// - Host data becomes a hipsr.compute holding one tensor.extract per output
//   element. Host data here is a shape vector, so a kernel would first have to
//   copy it to the device. Naming every read position needs constant indices
//   and a static data shape, which is what `Gather(Shape(x), const)` has.
//
// The result type comes from the operands, not the type ONNX declared: `data`
// gives the element type, the memory space and every axis but the gathered
// one, which the whole indices shape replaces. ONNX also lets `axis` count
// back from the end, so the conversion resolves it.
//
// Device data, an embedding lookup. Before:
//   %e = "onnx.Gather"(%table, %ids) {axis = 0 : si64}
//       : (tensor<8x4096xf16, #hipsr.mem<device>>,
//          tensor<?x?xi64, #hipsr.mem<device>>) -> tensor<?x?x4096xf16>
//
// After, with the ins types left out:
//   %init = hipsr.placeholder(%ctx) ins(%table, %ids)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<?x?x4096xf16, #hipsr.mem<device>>
//   %e = hipsr.gather(%ctx) ins(%table, %ids)
//       outs(%init : tensor<?x?x4096xf16, #hipsr.mem<device>>)
//       {axis = 0 : i64} : tensor<?x?x4096xf16, #hipsr.mem<device>>
//
// Host data, reading the leading dimension out of a shape. Before:
//   %zero = arith.constant dense<0> : tensor<i64>
//   %n = "onnx.Gather"(%shape, %zero) {axis = 0 : si64}
//       : (tensor<2xi64, #hipsr.mem<host>>, tensor<i64>) -> tensor<i64>
//
// After, with the region types left out:
//   %init = hipsr.placeholder(%ctx) ins(%shape)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<i64, #hipsr.mem<host>> shape_region {
//   ^bb0(%shape_shape):
//     %s = shape.const_shape []
//     hipsr.shape_yield %s
//   }
//   %n = hipsr.compute(%ctx) ins(%shape) outs(%init) {
//   ^bb0(%body_ctx, %in, %dest):
//     %p = arith.constant 0 : index
//     %e = tensor.extract %in[%p]
//     %out = tensor.from_elements %e
//     hipsr.compute_yield %out
//   } : tensor<i64, #hipsr.mem<host>>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"
#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace mlir {
namespace hipsr {

// Defined with the hipsr.gather verifier, which checks the same rule.
SmallVector<int64_t> inferGatherResultShape(ArrayRef<int64_t> dataShape,
                                            ArrayRef<int64_t> indicesShape,
                                            int64_t axis);

namespace {

//===----------------------------------------------------------------------===//
// The read positions
//===----------------------------------------------------------------------===//

// One data position per output element, in the row-major order
// tensor.from_elements takes them in. The reads run over `data` as
// [outer, axis, inner]: an output element keeps its outer and inner position
// and takes its axis position from the indices.
SmallVector<SmallVector<int64_t>> readPositions(ArrayRef<int64_t> dataShape,
                                                ArrayRef<int64_t> indexValues,
                                                int64_t axis) {
  int64_t axisSize = dataShape[axis];
  int64_t outerCount = ShapedType::getNumElements(dataShape.take_front(axis));
  int64_t innerCount =
      ShapedType::getNumElements(dataShape.drop_front(axis + 1));
  SmallVector<int64_t> dataStrides = computeStrides(dataShape);

  SmallVector<SmallVector<int64_t>> positions;
  positions.reserve(outerCount * indexValues.size() * innerCount);
  for (int64_t outer : llvm::seq(outerCount)) {
    for (int64_t index : indexValues) {
      for (int64_t inner : llvm::seq(innerCount)) {
        int64_t element = (outer * axisSize + index) * innerCount + inner;
        positions.push_back(delinearize(element, dataStrides));
      }
    }
  }
  return positions;
}

//===----------------------------------------------------------------------===//
// The host form
//===----------------------------------------------------------------------===//

// The result shape is fixed at compile time, so the region is a constant and
// needs no shape-graph inputs.
void populateHostShapeRegion(OpBuilder &builder, PlaceholderOp placeholder,
                             ArrayRef<int64_t> resultShape) {
  OpBuilder::InsertionGuard guard(builder);
  Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  Location loc = placeholder.getLoc();
  Value shape = shape::ConstShapeOp::create(
      builder, loc, shape::ShapeType::get(builder.getContext()),
      builder.getIndexTensorAttr(resultShape));
  ShapeYieldOp::create(builder, loc, ValueRange{shape});
}

// Every read position is known, so the body is a flat list of extracts rather
// than a loop.
void populateHostBody(OpBuilder &builder, ComputeOp computeOp,
                      RankedTensorType resultType,
                      ArrayRef<SmallVector<int64_t>> positions) {
  OpBuilder::InsertionGuard guard(builder);
  Location loc = computeOp.getLoc();

  // The body's arguments are the operands: ctx, then the inputs, then the
  // outputs.
  TypeRange argTypes = computeOp->getOperandTypes();
  SmallVector<Location> argLocs(argTypes.size(), loc);
  Block *body =
      builder.createBlock(&computeOp.getBody(), {}, argTypes, argLocs);
  builder.setInsertionPointToStart(body);

  // Neighbouring positions share most of their subscripts, and no CSE follows
  // the hipsr pipeline, so each value gets one constant.
  DenseMap<int64_t, Value> indexConstants;
  auto indexConstant = [&](int64_t index) -> Value {
    Value &constant = indexConstants[index];
    if (!constant) {
      constant = arith::ConstantIndexOp::create(builder, loc, index);
    }
    return constant;
  };

  Value data = body->getArgument(1);
  SmallVector<Value> elements =
      llvm::map_to_vector(positions, [&](ArrayRef<int64_t> position) -> Value {
        SmallVector<Value> subscript =
            llvm::map_to_vector(position, indexConstant);
        return tensor::ExtractOp::create(builder, loc, data, subscript);
      });
  Value result =
      tensor::FromElementsOp::create(builder, loc, resultType, elements);
  ComputeYieldOp::create(builder, loc, ValueRange{result});
}

LogicalResult replaceWithHostCompute(onnx::GatherOp op, Value data,
                                     RankedTensorType dataType, Value indices,
                                     RankedTensorType resultType, int64_t axis,
                                     Value ctx,
                                     ConversionPatternRewriter &rewriter) {
  // Unrolling the reads needs every position and every dimension at compile
  // time.
  DenseIntElementsAttr folded;
  if (!dataType.hasStaticShape() || !resultType.hasStaticShape() ||
      !matchPattern(indices, m_Constant(&folded))) {
    return rewriter.notifyMatchFailure(
        op, "expected static shapes and constant indices");
  }
  // tensor.from_elements writes at least one element.
  if (resultType.getNumElements() == 0) {
    return rewriter.notifyMatchFailure(op, "expected a non-empty result");
  }

  // ONNX lets an index count back from the end of the axis.
  int64_t axisSize = dataType.getDimSize(axis);
  SmallVector<int64_t> indexValues = llvm::map_to_vector(
      folded.getValues<APInt>(), [&](const APInt &value) -> int64_t {
        int64_t index = value.getSExtValue();
        return index < 0 ? index + axisSize : index;
      });
  if (!llvm::all_of(indexValues, [&](int64_t index) {
        return index >= 0 && index < axisSize;
      })) {
    return rewriter.notifyMatchFailure(op,
                                       "expected every index within the axis");
  }

  // Constant indices fix the shape; the data is there to match the compute.
  Location loc = op.getLoc();
  auto init = PlaceholderOp::create(rewriter, loc, TypeRange{resultType}, ctx,
                                    ValueRange{data}, PlaceholderType::Normal);
  populateHostShapeRegion(rewriter, init, resultType.getShape());

  auto computeOp =
      ComputeOp::create(rewriter, loc, TypeRange{resultType}, ctx,
                        ValueRange{data}, ValueRange{init.getResult(0)});
  populateHostBody(rewriter, computeOp, resultType,
                   readPositions(dataType.getShape(), indexValues, axis));
  rewriter.replaceOp(op, computeOp.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// The pattern
//===----------------------------------------------------------------------===//

struct GatherToHipsr : public OpConversionPattern<onnx::GatherOp> {
  // The type converter is unused: converting the operands would overwrite the
  // memory space their producers chose, which is what picks the form below. It
  // stays in the signature so every pattern is built the same way.
  GatherToHipsr(const TypeConverter &, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();
    auto dataType = dyn_cast<RankedTensorType>(data.getType());
    auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
    if (!dataType || !indicesType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor operands");
    }
    int64_t axis = op.getAxis();
    if (axis < 0) {
      axis += dataType.getRank();
    }
    if (axis < 0 || axis >= dataType.getRank()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected an axis within the data");
    }
    FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
    if (failed(ctx)) {
      return failure();
    }
    auto resultType = RankedTensorType::get(
        inferGatherResultShape(dataType.getShape(), indicesType.getShape(),
                               axis),
        dataType.getElementType(), dataType.getEncoding());

    if (isHostRankedTensor(dataType)) {
      return replaceWithHostCompute(op, data, dataType, indices, resultType,
                                    axis, *ctx, rewriter);
    }
    if (!isDeviceRankedTensor(dataType) || !isDeviceRankedTensor(indicesType)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected device-resident operands");
    }

    Location loc = op.getLoc();
    Value init = PlaceholderOp::create(rewriter, loc, TypeRange{resultType},
                                       *ctx, ValueRange{data, indices},
                                       PlaceholderType::Normal)
                     .getResult(0);
    auto gatherOp =
        GatherOp::create(rewriter, loc, TypeRange{resultType}, *ctx, data,
                         indices, init, rewriter.getI64IntegerAttr(axis));
    rewriter.replaceOp(op, gatherOp.getResult(0));
    return success();
  }
};

} // namespace

void populateGatherConversionPatterns(const TypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<GatherToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
