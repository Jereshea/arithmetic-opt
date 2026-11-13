// include "/home/mcw/mlir_assignment/llvm-project/mlir/include/mlir/IR/OpBase.td"
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
  let description = "Takes 2 integers and performs addition";
  let arguments = (ins AnyTypeOf<[I32, I32Tensor]>:$val1,
                       AnyTypeOf<[I32, I32Tensor]>:$val2);
  let results = (outs AnyTypeOf<[I32, I32Tensor]>:$output);
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
  let description = "Takes 2 integers and performs addition";
  let arguments = (ins AnyTypeOf<[I32, I32Tensor]>:$val1,
                       AnyTypeOf<[I32, I32Tensor]>:$val2);
  let results = (outs AnyTypeOf<[I32, I32Tensor]>:$output);
  let assemblyFormat = "$val1 `*` $val2 attr-dict `:` type($output)";
}

def mulbroadcasting_op : Op<arithmeticops,"mul_broadcasting_op">{
  let summary = "Performs Multiplication";
  let description = "Takes 2 integers and performs addition (broadcasting applies for different dtypes)";
  let arguments = (ins AnyTypeOf<[I32, I32Tensor]>:$val1,
                       AnyTypeOf<[I32, I32Tensor]>:$val2);
  let results = (outs AnyTypeOf<[I32, I32Tensor]>:$output);
  let assemblyFormat = "$val1 `*` $val2 attr-dict `:` type($val1) `,` type($val2) `->` type($output)";
}