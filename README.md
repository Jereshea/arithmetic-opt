# arithmetic-opt (`arithmeticops` MLIR Dialect)

A custom out-of-tree MLIR dialect (`arithmeticops`) and standalone optimizer driver (`arithmetic-opt`) built to explore MLIR Operation Definition Specification (ODS) TableGen modeling, in-place constant/identity folding (`hasFolder = 1`), structural canonicalization rewrite patterns (`hasCanonicalizer = 1` / `--canonicalize`), Dialect Conversion lowering passes (`--convert-arithmeticops-to-arith`), and `llvm-lit` / `FileCheck` regression testing.

> New to MLIR? See **[GUIDE.md](GUIDE.md)** for a step-by-step walkthrough of TableGen (`mlir-tblgen`), C++ dialect/op registration, folding, canonicalization, lowering passes, and CMake setup.

---

## Project Structure

```text
arithmetic-opt/
├── CMakeLists.txt                  # Top-level CMake configuration
├── GUIDE.md                        # Step-by-step guide for building, canonicalizing, and lowering a custom MLIR dialect
├── README.md                       # Project overview, operation summary, and usage
├── archive/                        # Archived early TableGen prototypes & parse-only tests (excluded from active build)
│   ├── arithmeticops_full.td       # Original ODS with broadcasting & WeightedAddOp prototypes
│   ├── add_broadcasting.mlir       # Scalar-to-tensor addition broadcast parse/verify test
│   ├── mul_broadcasting.mlir       # Scalar-to-3D-tensor multiplication broadcast test
│   └── weighted_add.mlir           # Optional operand & F32ArrayAttr weighted addition test
├── include/
│   └── arithmeticops/
│       ├── arithmetic.h            # Dialect declaration
│       ├── arithmeticop.h          # Operation class declarations & BytecodeOpInterface include
│       ├── arithmeticops.td        # Active TableGen dialect & operation definitions (addop, mulop, addcmulop)
│       ├── ArithmeticOpsLowering.h # Conversion pattern population declaration
│       ├── Passes.h                # Pass declarations & registration hooks
│       ├── Passes.td               # TableGen pass definition (--convert-arithmeticops-to-arith)
│       └── CMakeLists.txt          # mlir-tblgen code generation targets
├── lib/
│   └── arithmeticops/
│       ├── arithmetic.cpp          # Dialect initialization & registration (with Arith/Tensor dependentDialects)
│       ├── arithmeticop.cpp        # Operation definitions, fold() methods, and OpRewritePatterns
│       ├── ArithmeticOpsLowering.cpp # OpConversionPatterns (addop, mulop, addcmulop -> arith & tensor dialects)
│       ├── Passes.cpp              # ConvertArithmeticOpsToArithPass implementation
│       └── CMakeLists.txt          # MLIRArithmeticOps static library target
├── lit_tests/
│   ├── CMakeLists.txt              # Configures lit.site.cfg.py and check-arithmetic target
│   ├── lit.cfg.py                  # llvm-lit suite configuration (ShTest & tool substitutions)
│   ├── lit.site.cfg.py.in          # CMake template for build-time paths
│   ├── add.mlir                    # FileCheck regression tests for scalar/tensor add lowering
│   ├── mul.mlir                    # FileCheck regression tests for scalar/tensor mul lowering
│   ├── addcmul.mlir                # FileCheck regression tests for tensor addcmul lowering
│   ├── canonicalize_add.mlir       # FileCheck tests for addop folding (x + 0) & AddSelf (x + x -> x * 2)
│   ├── canonicalize_mul.mlir       # FileCheck tests for mulop folding (x * 1 -> x, x * 0 -> 0)
│   └── canonicalize_addcmul.mlir   # FileCheck tests for addcmulop folding & structural canonicalization
└── tools/
    ├── arithmetic-opt.cpp          # Standalone MLIR optimizer driver
    └── CMakeLists.txt              # arithmetic-opt executable target
```

---

## Active Operations, Folding & Canonicalization

The active build defines **`addop`**, **`mulop`**, and **`addcmulop`**, with `dependentDialects = ["mlir::arith::ArithDialect", "mlir::tensor::TensorDialect"]` declared on the dialect so `--canonicalize` and lowering passes can materialize `arith` and `tensor` operations seamlessly.

Canonicalization uses a **hybrid `fold()` + `OpRewritePattern` architecture**:
* **`fold()` (`hasFolder = 1`):** Handles in-place identity and zero simplifications that return an existing SSA operand without creating new operations.
* **`OpRewritePattern` (`hasCanonicalizer = 1`):** Handles structural transformations that replace an operation with newly constructed operations (e.g., `arith.constant`, `tensor.splat`, `arithmeticops.mul`, `arithmeticops.add`).

| Operation | Mnemonic | Traits & Types | `fold()` Rules (In-Place) | `OpRewritePattern` Rules (`--canonicalize`) | Lowering (`--convert-arithmeticops-to-arith`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`addop`** | `arithmeticops.add` | `[Pure, SameOperandsAndResultType]`<br>`I32, F32, I32Tensor, F32Tensor` | • `x + 0 -> x`<br>• `0 + x -> x` | • **`AddSelf`**: `x + x -> arithmeticops.mul(x, 2)` | Lowered to `arith.addi` (integers) or `arith.addf` (floats) |
| **`mulop`** | `arithmeticops.mul` | `[Pure, SameOperandsAndResultType]`<br>`I32, F32, I32Tensor, F32Tensor` | • `x * 1 -> x`, `1 * x -> x`<br>• `x * 0 -> 0`, `0 * x -> 0` | *(Handled via `fold()`)* | Lowered to `arith.muli` (integers) or `arith.mulf` (floats) |
| **`addcmulop`** | `arithmeticops.addcmul`<br>`input + value * (tensor1 * tensor2)` | `[Pure, AllShapesMatch, AllElementTypesMatch]`<br>`input, tensor1, tensor2`: `I32Tensor, F32Tensor`<br>`value`: `I32, F32` | • `input + 0 * (tensor1 * tensor2) -> input`<br>• `input + value * (dense<0> * tensor2) -> input`<br>• `input + value * (tensor1 * dense<0>) -> input` | • **`Input_Zero`**: `dense<0> + value * (tensor1 * tensor2) -> splat(value) * (tensor1 * tensor2)`<br>• **`Value_One`**: `input + 1 * (tensor1 * tensor2) -> input + (tensor1 * tensor2)`<br>• **`Tensor1_One`**: `input + value * (dense<1> * tensor2) -> input + (splat(value) * tensor2)`<br>• **`Tensor2_One`**: `input + value * (tensor1 * dense<1>) -> input + (splat(value) * tensor1)` | Lowered to `tensor.splat` + `arith.muli`/`mulf` + `arith.addi`/`addf` |

> **Archived Prototypes (`archive/`):** Earlier experimental ODS operations (`add_broadcasting_op`, `mul_broadcasting_op`, and `WeightedAddOp`) and their parse-only `.mlir` files are preserved in `archive/` outside the active CMake build tree.

---

## Prerequisites

* **LLVM / MLIR** built from source or installed (with `MLIRConfig.cmake`, `LLVMConfig.cmake`, `llvm-lit`, and `FileCheck`)
* **CMake** (>= 3.20)
* **Ninja** or **Make** build system and a C++17-compatible compiler (`g++` or `clang++`)

---

## Build Instructions

From the repository root (`/home/mcw/mlir_assignment/arithmetic-opt`):

```bash
mkdir -p build && cd build

cmake .. \
  -DMLIR_DIR=/home/mcw/mlir/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/mcw/mlir/llvm-project/build/lib/cmake/llvm

cmake --build . --target arithmetic-opt
```

*(Note: Ensure `-DMLIR_DIR` and `-DLLVM_DIR` point to the LLVM/MLIR build containing `llvm-lit` and `FileCheck` so `lit.site.cfg.py` resolves `LLVM_TOOLS_BINARY_DIR` automatically.)*

---

## Running `lit` Regression Tests (`lit_tests/`)

Automated canonicalization and lowering tests live in `lit_tests/` and use `llvm-lit` + `FileCheck` to verify IR transformations across all 6 test files (`add`, `mul`, `addcmul` lowering and canonicalization).

### 1. Build and Run the Full `lit` Test Suite via CMake

From inside `build/` (automatically recompiles `arithmetic-opt` if C++ files changed):

```bash
cmake --build . --target check-arithmetic
```

### 2. Run `llvm-lit` Directly (Fast Iteration on `.mlir` Tests)

Once `build/lit_tests/lit.site.cfg.py` has been generated by CMake, you can invoke `llvm-lit` directly from `build/` without triggering a CMake rebuild:

```bash
# Run all 6 tests in lit_tests/ (lowering + canonicalization) with verbose output
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/

# Run individual canonicalization or lowering test files
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/canonicalize_add.mlir
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/canonicalize_mul.mlir
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/canonicalize_addcmul.mlir
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/add.mlir
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/mul.mlir
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/addcmul.mlir
```

### 3. Run an Individual `FileCheck` Pipeline Manually

```bash
./bin/arithmetic-opt ../lit_tests/canonicalize_addcmul.mlir --canonicalize \
  | /home/mcw/mlir/llvm-project/build/bin/FileCheck ../lit_tests/canonicalize_addcmul.mlir
```

---

## Running Pass Pipelines on the CLI

### 1. Run Canonicalization (`--canonicalize`)

Simplify algebraic identities and fold constant zero/one operands across `add`, `mul`, and `addcmul`:

```bash
./build/bin/arithmetic-opt lit_tests/canonicalize_add.mlir --canonicalize
./build/bin/arithmetic-opt lit_tests/canonicalize_mul.mlir --canonicalize
./build/bin/arithmetic-opt lit_tests/canonicalize_addcmul.mlir --canonicalize
```

### 2. Canonicalize and Lower `arithmeticops` to Standard `arith` & `tensor` Dialects

Run `--canonicalize` first to eliminate redundant operations, then convert remaining `arithmeticops.add`, `arithmeticops.mul`, and `arithmeticops.addcmul` operations into `arith` and `tensor` dialect ops:

```bash
./build/bin/arithmetic-opt lit_tests/canonicalize_addcmul.mlir \
  --canonicalize \
  --convert-arithmeticops-to-arith
```

### 3. Chain Built-In MLIR Passes to Lower Scalar Ops All the Way to LLVM Dialect

Because `arithmetic-opt` registers MLIR's standard passes, you can chain scalar operations directly into upstream LLVM lowering passes on the CLI:

```bash
./build/bin/arithmetic-opt lit_tests/mul.mlir \
  --canonicalize \
  --convert-arithmeticops-to-arith \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts
```