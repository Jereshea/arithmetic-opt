#include "arithmeticops/arithmeticop.h"
#include "arithmeticops/arithmetic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
using namespace mlir;
#define GET_OP_CLASSES
#include "arithmeticops/arithmeticops.cpp.inc"

/*Canonilcazaion using ReWrite Pattern
namespace{

// Canonicalize for arithmetic::addop
struct AddZero : public OpRewritePattern<arithmetic::addop> {
  using OpRewritePattern<arithmetic::addop>::OpRewritePattern;

  LogicalResult matchAndRewrite(arithmetic::addop op,
                                PatternRewriter &rewriter) const override {

    // Canonicalize x+0->x
    if (matchPattern(op.getVal2(), m_Zero()) ||
        matchPattern(op.getVal2(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getVal1());
      return success();
    }

    // Canonicalize 0+x->x
    if (matchPattern(op.getVal1(), m_Zero()) ||
        matchPattern(op.getVal1(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getVal2());
      return success();
    }

    return failure();
  }
};

struct AddSelf : public OpRewritePattern<arithmetic::addop> {
  using OpRewritePattern<arithmetic::addop>::OpRewritePattern;

  LogicalResult matchAndRewrite(arithmetic::addop op,
                                PatternRewriter &rewriter) const override {

    // Canonicalize x+x->x*2
    if (op.getVal1() == op.getVal2()){
      // Create constant 2 or 2.0 matching scalar or tensor type
      Location loc = op.getLoc();
      Type type = op.getType();
      Value two;

      // Check if scalar/tensor is a int; if float assigns to nullptr
      if (auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type))) {
        auto attr = rewriter.getIntegerAttr(intType, 2);

        // IntTensor
        if (auto shapedType = dyn_cast<ShapedType>(type))
            two = arith::ConstantOp::create(
              rewriter, loc, DenseElementsAttr::get(shapedType, attr));

        // IntScalar
        else
            two = arith::ConstantOp::create(rewriter, loc, attr);
      }else if (auto floatType =
dyn_cast<FloatType>(getElementTypeOrSelf(type))) { auto attr =
rewriter.getFloatAttr(floatType, 2.0);

        // FloatTensor
        if (auto shapedType = dyn_cast<ShapedType>(type))
          two = arith::ConstantOp::create(
              rewriter, loc, DenseElementsAttr::get(shapedType, attr));
        // FloatScalar
        else
          two = arith::ConstantOp::create(rewriter, loc, attr);
        } else {
            return failure();
        }

        // Apply custom dialect mulop
        rewriter.replaceOpWithNewOp<arithmetic::mulop>(op, type, op.getVal1(),
two); return success();
    }

    return failure();
  }
};

// Canonicalize for arithmetic::mulop
struct MulZero : public OpRewritePattern<arithmetic::mulop> {
  using OpRewritePattern<arithmetic::mulop>::OpRewritePattern;

  LogicalResult matchAndRewrite(arithmetic::mulop op,
                                PatternRewriter &rewriter) const override {
    // Canonicalize x*0->0
    if (matchPattern(op.getVal2(), m_Zero()) ||
        matchPattern(op.getVal2(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getVal2());
      return success();
    }

    // Canonicalize 0*x->0
    if (matchPattern(op.getVal1(), m_Zero()) ||
        matchPattern(op.getVal1(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getVal1());
      return success();
    }

    return failure();
  }
};

struct MulOne : public OpRewritePattern<arithmetic::mulop> {
  using OpRewritePattern<arithmetic::mulop>::OpRewritePattern;

  LogicalResult matchAndRewrite(arithmetic::mulop op,
                                PatternRewriter &rewriter) const override {

    // Canonicalize x*1->x
    if (matchPattern(op.getVal2(), m_One()) ||
        matchPattern(op.getVal2(), m_OneFloat())) {
      rewriter.replaceOp(op, op.getVal1());
      return success();
    }

    // Canonicalize 1*x->x
    if (matchPattern(op.getVal1(), m_One()) ||
        matchPattern(op.getVal1(), m_OneFloat())) {
      rewriter.replaceOp(op, op.getVal2());
      return success();
    }

    return failure();
  }
};

}

void arithmetic::addop::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
    patterns.add<AddZero, AddSelf>(context);
}

void arithmetic::mulop::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
    patterns.add<MulZero, MulOne>(context);
}

*/

/* Canonilcazaion using ReWrite Pattern and Fold Method */

namespace {
struct AddSelf : public OpRewritePattern<arithmetic::addop> {
  using OpRewritePattern<arithmetic::addop>::OpRewritePattern;

  LogicalResult matchAndRewrite(arithmetic::addop op,
                                PatternRewriter &rewriter) const override {

    // Canonicalize x+x->x*2
    if (op.getVal1() == op.getVal2()) {
      // Create constant 2 or 2.0 matching scalar or tensor type
      Location loc = op.getLoc();
      Type type = op.getType();
      Value two;

      // Check if scalar/tensor is a int; if float assigns to nullptr
      if (auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type))) {
        auto attr = rewriter.getIntegerAttr(intType, 2);

        // IntTensor
        if (auto shapedType = dyn_cast<ShapedType>(type))
          two = arith::ConstantOp::create(
              rewriter, loc, DenseElementsAttr::get(shapedType, attr));

        // IntScalar
        else
          two = arith::ConstantOp::create(rewriter, loc, attr);
      } else if (auto floatType =
                     dyn_cast<FloatType>(getElementTypeOrSelf(type))) {
        auto attr = rewriter.getFloatAttr(floatType, 2.0);

        // FloatTensor
        if (auto shapedType = dyn_cast<ShapedType>(type))
          two = arith::ConstantOp::create(
              rewriter, loc, DenseElementsAttr::get(shapedType, attr));
        // FloatScalar
        else
          two = arith::ConstantOp::create(rewriter, loc, attr);
      } else {
        return failure();
      }

      // Apply custom dialect mulop
      rewriter.replaceOpWithNewOp<arithmetic::mulop>(op, type, op.getVal1(),
                                                     two);
      return success();
    }

    return failure();
  }
};

// addcmulop: input + value * (tensor1 * tensor2)
struct Input_Zero : public OpRewritePattern<arithmetic::addcmulop> {
  using OpRewritePattern<arithmetic::addcmulop>::OpRewritePattern;

  LogicalResult matchAndRewrite(arithmetic::addcmulop op,
                                PatternRewriter &rewriter) const override {

    // Canonicalize 0 + value * (tensor1 * tensor2)->value * (tensor1 * tensor2)
    if (matchPattern(op.getInput(), m_Zero()) ||
        matchPattern(op.getInput(), m_AnyZeroFloat())) {

      // Map to arithmetic::mulop
      Location loc = op.getLoc();
      Type tensorType = op.getType();
      Value splatVal =
          tensor::SplatOp::create(rewriter, loc, tensorType, op.getValue());
      Value tensors_prod = arithmetic::mulop::create(
          rewriter, loc, tensorType, op.getTensor1(), op.getTensor2());
      rewriter.replaceOpWithNewOp<arithmetic::mulop>(op, tensorType, splatVal,
                                                     tensors_prod);
      return success();
    }

    return failure();
  }
};

struct Value_One : public OpRewritePattern<arithmetic::addcmulop> {
  using OpRewritePattern<arithmetic::addcmulop>::OpRewritePattern;

  LogicalResult matchAndRewrite(arithmetic::addcmulop op,
                                PatternRewriter &rewriter) const override {

    // Canonicalize input + 1 * (tensor1 * tensor2)->input + (tensor1 * tensor2)
    if (matchPattern(op.getValue(), m_One()) ||
        matchPattern(op.getValue(), m_OneFloat())) {

      // Map to arithmetic::mulop
      Location loc = op.getLoc();
      Type tensorType = op.getType();
      Value tensors_prod = arithmetic::mulop::create(
          rewriter, loc, tensorType, op.getTensor1(), op.getTensor2());
      rewriter.replaceOpWithNewOp<arithmetic::addop>(
          op, tensorType, op.getInput(), tensors_prod);
      return success();
    }

    return failure();
  }
};

struct Tensor1_One : public OpRewritePattern<arithmetic::addcmulop> {
  using OpRewritePattern<arithmetic::addcmulop>::OpRewritePattern;

  LogicalResult matchAndRewrite(arithmetic::addcmulop op,
                                PatternRewriter &rewriter) const override {

    // Canonicalize input + value * (dense<1> * tensor2)->input + (value *
    // tensor1)
    if (matchPattern(op.getTensor1(), m_One()) ||
        matchPattern(op.getTensor1(), m_OneFloat())) {

      // Map to arithmetic::mulop
      Location loc = op.getLoc();
      Type tensorType = op.getType();
      Value splatVal =
          tensor::SplatOp::create(rewriter, loc, tensorType, op.getValue());
      Value tensor2_value__prod = arithmetic::mulop::create(
          rewriter, loc, tensorType, splatVal, op.getTensor2());
      rewriter.replaceOpWithNewOp<arithmetic::addop>(
          op, tensorType, op.getInput(), tensor2_value__prod);
      return success();
    }

    return failure();
  }
};

struct Tensor2_One : public OpRewritePattern<arithmetic::addcmulop> {
  using OpRewritePattern<arithmetic::addcmulop>::OpRewritePattern;

  LogicalResult matchAndRewrite(arithmetic::addcmulop op,
                                PatternRewriter &rewriter) const override {

    // Canonicalize input + value * (tensor1 * dense<1>)->input + (value *
    // tensor1)
    if (matchPattern(op.getTensor2(), m_One()) ||
        matchPattern(op.getTensor2(), m_OneFloat())) {

      // Map to arithmetic::mulop
      Location loc = op.getLoc();
      Type tensorType = op.getType();
      Value splatVal =
          tensor::SplatOp::create(rewriter, loc, tensorType, op.getValue());
      Value tensor1_value__prod = arithmetic::mulop::create(
          rewriter, loc, tensorType, splatVal, op.getTensor1());
      rewriter.replaceOpWithNewOp<arithmetic::addop>(
          op, tensorType, op.getInput(), tensor1_value__prod);
      return success();
    }

    return failure();
  }
};
} // namespace

OpFoldResult arithmetic::addop::fold(FoldAdaptor adaptor) {

  // Canonicalize x+0->x
  if (matchPattern(adaptor.getVal2(), m_Zero()) ||
      matchPattern(adaptor.getVal2(), m_AnyZeroFloat())) {
    return getVal1();
  }

  // Canonicalize 0+x->x
  if (matchPattern(adaptor.getVal1(), m_Zero()) ||
      matchPattern(adaptor.getVal1(), m_AnyZeroFloat())) {
    return getVal2();
  }

  return {};
}

OpFoldResult arithmetic::mulop::fold(FoldAdaptor adaptor) {

  // Canonicalize x*0->0
  if (matchPattern(adaptor.getVal2(), m_Zero()) ||
      matchPattern(adaptor.getVal2(), m_AnyZeroFloat())) {
    return getVal2();
  }

  // Canonicalize 0*x->0
  if (matchPattern(adaptor.getVal1(), m_Zero()) ||
      matchPattern(adaptor.getVal1(), m_AnyZeroFloat())) {
    return getVal1();
  }

  // Canonicalize x*1->x
  if (matchPattern(adaptor.getVal2(), m_One()) ||
      matchPattern(adaptor.getVal2(), m_OneFloat())) {
    return getVal1();
  }

  // Canonicalize 1*x->x
  if (matchPattern(adaptor.getVal1(), m_One()) ||
      matchPattern(adaptor.getVal1(), m_OneFloat())) {
    return getVal2();
  }
  return {};
}

OpFoldResult arithmetic::addcmulop::fold(FoldAdaptor adaptor) {

  // input + value * (tensor1 * tensor2) ; here if value/tensor1/tensor2 is 0,
  // the entire mul can be skipped Canonicalize input + 0 * (tensor1 *
  // tensor2)->input
  if (matchPattern(adaptor.getValue(), m_Zero()) ||
      matchPattern(adaptor.getValue(), m_AnyZeroFloat())) {
    return getInput();
  }

  // Canonicalize input + value * (tensor1<0> * tensor2)->input
  if (matchPattern(adaptor.getTensor1(), m_Zero()) ||
      matchPattern(adaptor.getTensor1(), m_AnyZeroFloat())) {
    return getInput();
  }

  // Canonicalize input + value * (tensor1 * tensor2<0>)->input
  if (matchPattern(adaptor.getTensor2(), m_Zero()) ||
      matchPattern(adaptor.getTensor2(), m_AnyZeroFloat())) {
    return getInput();
  }
  return {};
}

void arithmetic::addop::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add<AddSelf>(context);
}
void arithmetic::addcmulop::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<Input_Zero, Value_One, Tensor1_One, Tensor2_One>(context);
}