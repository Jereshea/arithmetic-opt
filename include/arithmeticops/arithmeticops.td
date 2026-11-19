include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def arithmeticops: Dialect{
    let name = "arithmeticops";
    let summary = "Performs Arithmetic Operations";
    let description = "When this dialect is called this helps in performing arithmetic ops: Add, Mul";
    let cppNamespace = "::arithmetic";

}

def addop : Op<arithmeticops,"add",[SameOperandsAndResultType]>{
  let summary = "Performs Addition";
  let description = "Takes 2 integers/float and performs addition";
  let arguments = (ins AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$val1,
                       AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$val2);
  let results = (outs AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$output);
}

def add_broadcasting_op : Op<arithmeticops,"add_broadcasting_op">{
  let summary = "Performs Addition";
  let description = "Takes 2 integers and performs addition (broadcasting applies for different dtypes)";
  let arguments = (ins AnyTypeOf<[I32, I32Tensor]>:$val1,
                       AnyTypeOf<[I32, I32Tensor]>:$val2);
  let results = (outs AnyTypeOf<[I32, I32Tensor]>:$output);
}

def mulop : Op<arithmeticops,"mul",[SameOperandsAndResultType]>{
  let summary = "Performs Multiplication";
  let description = "Takes 2 integers/float and performs Multiplication";
  let arguments = (ins AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$val1,
                       AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$val2);
  let results = (outs AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$output);
  let assemblyFormat = "$val1 `*` $val2 attr-dict `:` type($output)";
}

def mulbroadcasting_op : Op<arithmeticops,"mul_broadcasting_op">{
  let summary = "Performs Multiplication";
  let description = "Takes 2 integers and performs Multiplication (broadcasting applies for different dtypes)";
  let arguments = (ins AnyTypeOf<[I32, I32Tensor]>:$val1,
                       AnyTypeOf<[I32, I32Tensor]>:$val2);
  let results = (outs AnyTypeOf<[I32, I32Tensor]>:$output);
  let assemblyFormat = "$val1 `*` $val2 attr-dict `:` type($val1) `,` type($val2) `->` type($output)";
}

// Types: https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/CommonTypeConstraints.td 
// Attributes: https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/CommonAttrConstraints.td 
// NOTE: Checks individual operand/result types only; shape broadcasting and int-to-float promotion should be handled in lowering.
def WeightedAddOp : Op<arithmeticops, "weighted_add"> {
  let summary = "Adds an optional bias tensor and applies float scaling attributes";
  let description = [{ Takes a mandatory integer or float tensor, an optional second integer or float tensor, and a float array attribute `testAttr`, producing a 3D float tensor.}];
  let arguments = (ins TensorOf<[AnyInteger, AnyFloat]>:$input,Optional<TensorOf<[AnyInteger, AnyFloat]>>:$bias,F32ArrayAttr:$testAttr);
  let results = (outs 3DTensorOf<[AnyFloat]>:$output);
  let assemblyFormat = [{ $input (`,` $bias^)? `testAttr` `=` $testAttr attr-dict `:` type($input) (`,` type($bias)^)? `->` type($output)}];
}