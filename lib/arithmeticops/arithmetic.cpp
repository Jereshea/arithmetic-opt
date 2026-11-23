#include "arithmeticops/arithmetic.h"
#include "arithmeticops/arithmeticop.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace arithmetic;

#include "arithmeticops/arithmeticopsDialect.cpp.inc"

void arithmeticops::initialize() {
  addOperations<
#define GET_OP_LIST
#include "arithmeticops/arithmeticops.cpp.inc"
      >();
}