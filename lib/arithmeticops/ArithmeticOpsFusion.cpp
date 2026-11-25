#include "arithmeticops/ArithmeticOpsFusion.h"
#include "arithmeticops/arithmeticop.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

struct FuseAddMulPattern : public OpRewritePattern<arithmetic::mulop> {
  using OpRewritePattern<arithmetic::mulop>::OpRewritePattern;

  LogicalResult matchAndRewrite(arithmetic::mulop mulOp,
                                PatternRewriter &rewriter) const override {
    Value val1 = mulOp.getVal1();
    Value val2 = mulOp.getVal2();

    arithmetic::addop addOp = nullptr;
    Value mulOperand;

    // Check val1: (a + b) * c
    if (auto add_expression = val1.getDefiningOp<arithmetic::addop>()) {
      if (add_expression->hasOneUse()) {
        addOp = add_expression;
        mulOperand = val2;
      }
    }

    // Check val2 (commutativity): c * (a + b)
    if (!addOp) {
      if (auto add_expression = val2.getDefiningOp<arithmetic::addop>()) {
        if (add_expression->hasOneUse()) {
          addOp = add_expression;
          mulOperand = val1;
        }
      }
    }

    if (!addOp)
      return failure();

    rewriter.replaceOpWithNewOp<arithmetic::fusedAddMulop>(
        mulOp, mulOp.getType(), addOp.getVal1(), addOp.getVal2(), mulOperand);
    return success();
  }
};

} // namespace

void arithmetic::populateArithmeticOpsFusionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FuseAddMulPattern>(patterns.getContext());
}
