// Reference:
// https://github.com/llvm/torch-mlir/blob/main/lib/Conversion/TorchToStablehlo/Basic.cpp

#include "arithmeticops/ArithmeticOpsLowering.h"
#include "arithmeticops/arithmeticop.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct AddOpConverter : public OpConversionPattern<arithmetic::addop> {
  using OpConversionPattern<arithmetic::addop>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arithmetic::addop op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getVal1(),
                                               adaptor.getVal2());
    return success();
  }
};

struct MulOpConverter : public OpConversionPattern<arithmetic::mulop> {
  using OpConversionPattern<arithmetic::mulop>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arithmetic::mulop op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::MulIOp>(op, adaptor.getVal1(),
                                               adaptor.getVal2());
    return success();
  }
};

} // namespace

void arithmetic::populateArithmeticOpsToArithConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AddOpConverter>(patterns.getContext());
  patterns.add<MulOpConverter>(patterns.getContext());
}