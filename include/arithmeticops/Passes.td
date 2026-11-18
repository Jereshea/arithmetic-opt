// Reference: https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Pass/PassBase.td 
#ifndef ARITHMETICOPS_PASSES
#define ARITHMETICOPS_PASSES

include "mlir/Pass/PassBase.td"

def ConvertArithmeticOpsToArith : Pass<"convert-arithmeticops-to-arith", "mlir::ModuleOp"> {
  let summary = "Lower arithmeticops dialect operations to standard Arith dialect";
  let description = [{Converts custom operations in the `arithmeticops` dialect (such as `add` and `mul`) into standard `arith` dialect operations.}];
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif 