# arithmetic-opt (`arithmeticops` MLIR Dialect)

A custom out-of-tree MLIR dialect (`arithmeticops`) and standalone optimizer driver (`arithmetic-opt`) built to explore MLIR Operation Definition Specification (ODS) TableGen modeling, traits, custom assembly formats, optional operands, attributes, and Dialect Conversion lowering passes (`--convert-arithmeticops-to-arith`).

> New to MLIR? See **[GUIDE.md](GUIDE.md)** for a step-by-step walkthrough of TableGen (`mlir-tblgen`), C++ dialect/op registration, lowering passes, and CMake setup.

---

## Project Structure

```text
arithmetic-opt/
├── CMakeLists.txt                  # Top-level CMake configuration
├── GUIDE.md                        # Step-by-step guide for building and lowering a custom MLIR dialect
├── README.md                       # Project overview, operation summary, and usage
├── include/
│   └── arithmeticops/
│       ├── arithmetic.h            # Dialect declaration
│       ├── arithmeticop.h          # Operation class declarations & BytecodeOpInterface include
│       ├── arithmeticops.td        # TableGen dialect & operation definitions
│       ├── ArithmeticOpsLowering.h # Conversion pattern population declaration
│       ├── Passes.h                # Pass declarations & registration hooks
│       ├── Passes.td               # TableGen pass definition (--convert-arithmeticops-to-arith)
│       └── CMakeLists.txt          # mlir-tblgen code generation targets
├── lib/
│   └── arithmeticops/
│       ├── arithmetic.cpp          # Dialect initialization & registration
│       ├── arithmeticop.cpp        # Generated operation definitions
│       ├── ArithmeticOpsLowering.cpp # OpConversionPatterns (addop -> arith.addi, mulop -> arith.muli)
│       ├── Passes.cpp              # ConvertArithmeticOpsToArithPass implementation
│       └── CMakeLists.txt          # MLIRArithmeticOps static library target
├── tools/
│   ├── arithmetic-opt.cpp          # Standalone MLIR optimizer driver
│   └── CMakeLists.txt              # arithmetic-opt executable target
└── test/
    ├── add.mlir                    # Scalar element-wise addition test
    ├── add_broadcasting.mlir       # Scalar-to-tensor addition broadcast test
    ├── mul.mlir                    # Constant scalar multiplication test
    ├── mul_generic.mlir            # Generic syntax multiplication round-trip test
    ├── mul_broadcasting.mlir       # Scalar-to-3D-tensor multiplication broadcast test
    └── weighted_add.mlir           # Optional operand & F32ArrayAttr weighted addition test
```

---

## Supported Operations & Design Rationale

Each operation in `arithmeticops` isolates a specific MLIR TableGen / ODS concept:

| Operation | Mnemonic | Traits / Constraints | Assembly Format | Lowering Status | Why It Was Implemented |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`addop`** | `arithmeticops.add` | `[SameOperandsAndResultType]` over `AnyTypeOf<[I32, I32Tensor]>` | Generic quoted syntax | Lowered to `arith.addi` via `--convert-arithmeticops-to-arith` | Demonstrates built-in MLIR traits that automatically enforce identical types across all operands and results without writing custom verifiers. |
| **`add_broadcasting_op`** | `arithmeticops.add_broadcasting_op` | Unconstrained `AnyTypeOf<[I32, I32Tensor]>` per operand and result | Generic quoted syntax | Parse/verify only | Implemented to study how relaxing `SameOperandsAndResultType` allows mixed scalar/tensor signatures at the ODS level, separating structural IR parsing from actual broadcasting math. |
| **`mulop`** | `arithmeticops.mul` | `[SameOperandsAndResultType]` over `AnyTypeOf<[I32, I32Tensor]>` | `%lhs * %rhs : type` | Lowered to `arith.muli` via `--convert-arithmeticops-to-arith` | Demonstrates custom declarative `assemblyFormat` where a single trailing type (`type($output)`) is sufficient because all operand and result types match. |
| **`mulbroadcasting_op`** | `arithmeticops.mul_broadcasting_op` | Mixed `AnyTypeOf<[I32, I32Tensor]>` operands and result | `%lhs * %rhs : lhs_type, rhs_type -> res_type` | Parse/verify only | Demonstrates why custom assembly formats require explicit functional type signatures (`type($val1), type($val2) -> type($output)`) once operand and result types are allowed to differ for broadcasting. |
| **`WeightedAddOp`** | `arithmeticops.weighted_add` | Mandatory `TensorOf<[AnyInteger, AnyFloat]>` `$input`, `Optional<TensorOf<[AnyInteger, AnyFloat]>>` `$bias`, `F32ArrayAttr` `$testAttr`, and `3DTensorOf<[AnyFloat]>` `$output` | `$input (, $bias)? testAttr = $testAttr : type($input) (, type($bias))? -> type($output)` | Parse/verify only | Combines runtime optional operands (`Optional<...>`), compile-time array attributes (`F32ArrayAttr`), mixed integer/float tensor inputs, and fixed-rank output constraints (`3DTensorOf<[AnyFloat]>`) in a single operation. |

---

## Prerequisites

* **LLVM / MLIR** built and installed (with `MLIRConfig.cmake` and `LLVMConfig.cmake` available)
* **CMake** (>= 3.20)
* **Ninja** build system and a C++17-compatible compiler (`g++` or `clang++`)

---

## Build Instructions

From the repository root (`/home/mcw/mlir_assignment/arithmetic-opt`):

```bash
mkdir -p build && cd build

cmake -G Ninja .. \
  -DMLIR_DIR=/usr/local/lib/cmake/mlir \
  -DLLVM_DIR=/usr/local/lib/cmake/llvm

ninja arithmetic-opt
```

*(Note: If your LLVM/MLIR installation lives in a custom build directory rather than `/usr/local`, pass `-DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir` and `-DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm`.)*

---

## Running Tests & Lowering Passes

### 1. Parse and Verify Custom Dialect IR

Run `arithmetic-opt` on an individual `.mlir` test file:

```bash
./build/bin/arithmetic-opt test/weighted_add.mlir
```

Run the entire parse/verify test suite in `test/`:

```bash
for f in test/*.mlir; do
  echo "=== Testing $f ==="
  ./build/bin/arithmetic-opt "$f"
done
```

### 2. Lower `arithmeticops` to Standard `arith` Dialect

Convert `arithmeticops.add` and `arithmeticops.mul` into `arith.addi` and `arith.muli`:

```bash
./build/bin/arithmetic-opt test/add.mlir --convert-arithmeticops-to-arith
./build/bin/arithmetic-opt test/mul.mlir --convert-arithmeticops-to-arith
```

### 3. Chain Built-In MLIR Passes to Lower All the Way to LLVM Dialect

Because `arithmetic-opt` registers MLIR's standard passes, you can chain the output directly into upstream LLVM lowering passes on the CLI:

```bash
./build/bin/arithmetic-opt test/mul.mlir \
  --convert-arithmeticops-to-arith \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts
```

### Example IR (`test/weighted_add.mlir`)

```mlir
module {
  // Case 1: Mandatory integer tensor + optional float bias tensor + F32ArrayAttr -> 3D float tensor
  func.func @test_weighted_add_with_bias(%in: tensor<2x3x4xi32>, %bias: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = arithmeticops.weighted_add %in, %bias testAttr = [1.5 : f32, 2.0 : f32, 0.5 : f32]
        : tensor<2x3x4xi32>, tensor<2x3x4xf32> -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }

  // Case 2: Optional $bias omitted, 2D float tensor input -> 3D float tensor output
  func.func @test_weighted_add_without_bias(%in: tensor<3x4xf32>) -> tensor<2x3x4xf32> {
    %1 = arithmeticops.weighted_add %in testAttr = [1.0 : f32, 2.0 : f32, 3.0 : f32]
        : tensor<3x4xf32> -> tensor<2x3x4xf32>
    return %1 : tensor<2x3x4xf32>
  }
}
```