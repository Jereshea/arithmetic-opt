#include "arithmeticops/arithmeticop.h"
#include "arithmeticops/arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
using namespace mlir;
#define GET_OP_CLASSES
#include "arithmeticops/arithmeticops.cpp.inc"

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
      }else if (auto floatType = dyn_cast<FloatType>(getElementTypeOrSelf(type))) {
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
        rewriter.replaceOpWithNewOp<arithmetic::mulop>(op, type, op.getVal1(), two);
        return success();
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