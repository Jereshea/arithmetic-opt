// Reference:
// https://github.com/llvm/torch-mlir/blob/main/lib/Conversion/TorchToStablehlo/Basic.cpp

#include "arithmeticops/ArithmeticOpsLowering.h"
#include "arithmeticops/arithmeticop.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct AddOpConverter : public OpConversionPattern<arithmetic::addop> {
  using OpConversionPattern<arithmetic::addop>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arithmetic::addop op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type element_type = getElementTypeOrSelf(op.getType());

    if (isa<IntegerType>(element_type)) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getVal1(),
                                                 adaptor.getVal2());
      return success();
    }

    if (isa<FloatType>(element_type)) {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getVal1(),
                                                 adaptor.getVal2());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported element type");
  }
};

struct MulOpConverter : public OpConversionPattern<arithmetic::mulop> {
  using OpConversionPattern<arithmetic::mulop>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arithmetic::mulop op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type element_type = getElementTypeOrSelf(op.getType());

    if (isa<IntegerType>(element_type)) {
      rewriter.replaceOpWithNewOp<arith::MulIOp>(op, adaptor.getVal1(),
                                                 adaptor.getVal2());
      return success();
    }

    if (isa<FloatType>(element_type)) {
      rewriter.replaceOpWithNewOp<arith::MulFOp>(op, adaptor.getVal1(),
                                                 adaptor.getVal2());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported element type");
  }
};

struct AddcMulOpConverter : public OpConversionPattern<arithmetic::addcmulop> {
  using OpConversionPattern<arithmetic::addcmulop>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arithmetic::addcmulop op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto tensorType = cast<RankedTensorType>(op.getType());
    Type element_type = getElementTypeOrSelf(op.getType());

    // Splat scalar 'value' into a tensor of the same shape
    Value value_tensor =
        tensor::SplatOp::create(rewriter, loc, tensorType, adaptor.getValue());

    // input + value * (tensor1 * tensor2)
    if (isa<IntegerType>(element_type)) {
      Value tensors_product = arith::MulIOp::create(
          rewriter, loc, adaptor.getTensor1(), adaptor.getTensor2());
      Value value_tensors_product =
          arith::MulIOp::create(rewriter, loc, value_tensor, tensors_product);
      Value input_value_tensors_add = arith::AddIOp::create(
          rewriter, loc, adaptor.getInput(), value_tensors_product);

      rewriter.replaceOp(op, input_value_tensors_add);
      return success();
    }

    if (isa<FloatType>(element_type)) {
      Value tensors_product = arith::MulFOp::create(
          rewriter, loc, adaptor.getTensor1(), adaptor.getTensor2());
      Value value_tensors_product =
          arith::MulFOp::create(rewriter, loc, value_tensor, tensors_product);
      Value input_value_tensors_add = arith::AddFOp::create(
          rewriter, loc, adaptor.getInput(), value_tensors_product);

      rewriter.replaceOp(op, input_value_tensors_add);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported element type");
  }
};

struct FusedAddMulOpConverter
    : public OpConversionPattern<arithmetic::fusedAddMulop> {
  using OpConversionPattern<arithmetic::fusedAddMulop>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arithmetic::fusedAddMulop op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type element_type = getElementTypeOrSelf(op.getType());

    if (isa<IntegerType>(element_type)) {
      Value sum = arith::AddIOp::create(rewriter, loc, adaptor.getInput_1(),
                                        adaptor.getInput_2());
      rewriter.replaceOpWithNewOp<arith::MulIOp>(op, sum,
                                                 adaptor.getInput_3());
      return success();
    }

    if (isa<FloatType>(element_type)) {
      Value sum = arith::AddFOp::create(rewriter, loc, adaptor.getInput_1(),
                                        adaptor.getInput_2());
      rewriter.replaceOpWithNewOp<arith::MulFOp>(op, sum,
                                                 adaptor.getInput_3());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported element type");
  }
};

} // namespace

void arithmetic::populateArithmeticOpsToArithConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AddOpConverter>(patterns.getContext());
  patterns.add<MulOpConverter>(patterns.getContext());
  patterns.add<AddcMulOpConverter>(patterns.getContext());
  patterns.add<FusedAddMulOpConverter>(patterns.getContext());
}