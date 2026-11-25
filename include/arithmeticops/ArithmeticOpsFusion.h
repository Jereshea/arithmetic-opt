#ifndef ARITHMETICOPS_ARITHMETICOPSFUSION_H
#define ARITHMETICOPS_ARITHMETICOPSFUSION_H

#include "mlir/IR/PatternMatch.h"

namespace arithmetic {

/// Populates patterns to lower `arithmeticops` operations to the `arith`
/// dialect.
void populateArithmeticOpsFusionPatterns(
    mlir::RewritePatternSet &patterns);

} // namespace arithmetic

#endif