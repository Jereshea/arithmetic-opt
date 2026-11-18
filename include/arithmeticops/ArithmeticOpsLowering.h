#ifndef ARITHMETICOPS_ARITHMETICOPSLOWERING_H
#define ARITHMETICOPS_ARITHMETICOPSLOWERING_H

#include "mlir/IR/PatternMatch.h"

namespace arithmetic {

/// Populates patterns to lower `arithmeticops` operations to the `arith` dialect.
void populateArithmeticOpsToArithConversionPatterns(
    mlir::RewritePatternSet &patterns);

} 

#endif 