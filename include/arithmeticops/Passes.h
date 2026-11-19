#ifndef ARITHMETICOPS_PASSES_H
#define ARITHMETICOPS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace arithmetic {
#define GEN_PASS_DECL
#include "arithmeticops/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "arithmeticops/Passes.h.inc"
} // namespace arithmetic

#endif