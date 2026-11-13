#ifndef ARITHMETIC_OP_H
#define ARITHMETIC_OP_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#define GET_OP_CLASSES
#include "arithmeticops/arithmeticops.h.inc"

#endif