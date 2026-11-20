#ifndef ARITHMETIC_OP_H
#define ARITHMETIC_OP_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/TypeID.h"
#define GET_OP_CLASSES
#include "arithmeticops/arithmeticops.h.inc"

#endif