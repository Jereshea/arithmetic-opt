#include "arithmeticops/arithmetic.h"
#include "arithmeticops/arithmeticop.h"

using namespace mlir;
using namespace arithmetic;

#include "arithmeticops/arithmeticopsDialect.cpp.inc"

void arithmeticops::initialize() {
  addOperations<
#define GET_OP_LIST
#include "arithmeticops/arithmeticops.cpp.inc"
      >();
}