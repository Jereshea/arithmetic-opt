include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def arithmeticops: Dialect{
    let name = "arithmeticops";
    let summary = "Performs Arithmetic Operations";
    let description = "When this dialect is called this helps in performing arithmetic ops: Add, Mul";
    let cppNamespace = "::arithmetic";
    let dependentDialects = [
      "mlir::arith::ArithDialect",
      "mlir::tensor::TensorDialect"
    ];

}

def addop : Op<arithmeticops,"add",[Pure, SameOperandsAndResultType]>{
  let summary = "Performs Addition";
  let description = "Takes 2 integers/float and performs addition";
  let arguments = (ins AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$val1,
                       AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$val2);
  let results = (outs AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$output);
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def mulop : Op<arithmeticops,"mul",[Pure, SameOperandsAndResultType]>{
  let summary = "Performs Multiplication";
  let description = "Takes 2 integers/float and performs Multiplication";
  let arguments = (ins AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$val1,
                       AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$val2);
  let results = (outs AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$output);
  let assemblyFormat = "$val1 `*` $val2 attr-dict `:` type($output)";
  let hasFolder = 1;
}

def addcmulop : Op<arithmeticops,"addcmul",[Pure, AllShapesMatch<["input", "tensor1", "tensor2", "output"]>,AllElementTypesMatch<["input", "value", "tensor1", "tensor2", "output"]>]>{
  let summary = "Performs Fused add-constant-multiply: input + value * (tensor1 * tensor2)";
  let description = "Computes element-wise `input + value * (tensor1 * tensor2)` where `input`, `tensor1`, and `tensor2` are tensors of `i32` or `f32`, and `value` is a scalar multiplier of the matching element type.";
  let arguments = (ins AnyTypeOf<[I32Tensor, F32Tensor]>:$input,
                       AnyTypeOf<[I32, F32]>:$value,
                       AnyTypeOf<[I32Tensor, F32Tensor]>:$tensor1,
                       AnyTypeOf<[I32Tensor, F32Tensor]>:$tensor2);
  let results = (outs AnyTypeOf<[I32Tensor, F32Tensor]>:$output);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def fusedAddMulop : Op<arithmeticops,"fusedaddmul",[Pure, SameOperandsAndResultType]>{
  let summary = "Fused add followed by multiply: (a + b) * c";
  let description = "Computes if a= (b+c); d = a*e in a single fused operation.";
  let arguments = (ins AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$input_1,
                       AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$input_2,
                       AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$input_3);
  let results = (outs AnyTypeOf<[I32, F32, I32Tensor, F32Tensor]>:$output);
}