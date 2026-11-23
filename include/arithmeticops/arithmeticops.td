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
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def mulop : Op<arithmeticops,"mul",[SameOperandsAndResultType]>{
  let summary = "Performs Multiplication";
  let description = "Takes 2 integers/float and performs Multiplication";
  let arguments = (ins AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$val1,
                       AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$val2);
  let results = (outs AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$output);
  let assemblyFormat = "$val1 `*` $val2 attr-dict `:` type($output)";
  let hasFolder = 1;
}
