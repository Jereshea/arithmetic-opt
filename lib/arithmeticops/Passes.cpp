#include "arithmeticops/Passes.h"
#include "arithmeticops/ArithmeticOpsLowering.h"
#include "arithmeticops/arithmetic.h"
#include "arithmeticops/arithmeticop.h"
#include "arithmeticops/ArithmeticOpsFusion.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace arithmetic {
#define GEN_PASS_DEF_CONVERTARITHMETICOPSTOARITH
#define GEN_PASS_DEF_FUSEARITHMETICOPS
#include "arithmeticops/Passes.h.inc"
} // namespace arithmetic

using namespace mlir;

namespace {

struct ConvertArithmeticOpsToArithPass
    : public arithmetic::impl::ConvertArithmeticOpsToArithBase<
          ConvertArithmeticOpsToArithPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect>();
    target.addIllegalOp<arithmetic::addop, arithmetic::mulop,
                        arithmetic::addcmulop,
                        arithmetic::fusedAddMulop>();

    RewritePatternSet patterns(context);
    arithmetic::populateArithmeticOpsToArithConversionPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

struct FuseArithmeticOpsPass
    : public arithmetic::impl::FuseArithmeticOpsBase<FuseArithmeticOpsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    RewritePatternSet patterns(context);
    arithmetic::populateArithmeticOpsFusionPatterns(patterns);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace