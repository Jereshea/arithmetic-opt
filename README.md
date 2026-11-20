# arithmetic-opt (`arithmeticops` MLIR Dialect)

A custom out-of-tree MLIR dialect (`arithmeticops`) and standalone optimizer driver (`arithmetic-opt`) built to explore MLIR Operation Definition Specification (ODS) TableGen modeling, canonicalization rewrite patterns (`--canonicalize`), Dialect Conversion lowering passes (`--convert-arithmeticops-to-arith`), and `llvm-lit` / `FileCheck` regression testing.

> New to MLIR? See **[GUIDE.md](GUIDE.md)** for a step-by-step walkthrough of TableGen (`mlir-tblgen`), C++ dialect/op registration, canonicalization, lowering passes, and CMake setup.

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
│       ├── arithmeticops.td        # Active TableGen dialect & operation definitions (addop & mulop with canonicalizers)
│       ├── ArithmeticOpsLowering.h # Conversion pattern population declaration
│       ├── Passes.h                # Pass declarations & registration hooks
│       ├── Passes.td               # TableGen pass definition (--convert-arithmeticops-to-arith)
│       └── CMakeLists.txt          # mlir-tblgen code generation targets
├── lib/
│   └── arithmeticops/
│       ├── arithmetic.cpp          # Dialect initialization & registration
│       ├── arithmeticop.cpp        # Operation definitions & canonicalization patterns (AddZero, AddSelf, MulZero, MulOne)
│       ├── ArithmeticOpsLowering.cpp # OpConversionPatterns (addop -> arith.addi/addf, mulop -> arith.muli/mulf)
│       ├── Passes.cpp              # ConvertArithmeticOpsToArithPass implementation
│       └── CMakeLists.txt          # MLIRArithmeticOps static library target
├── lit_tests/
│   ├── CMakeLists.txt              # Configures lit.site.cfg.py and check-arithmetic target
│   ├── lit.cfg.py                  # llvm-lit suite configuration (ShTest & tool substitutions)
│   ├── lit.site.cfg.py.in          # CMake template for build-time paths
│   ├── add.mlir                    # FileCheck regression tests for scalar/tensor add lowering
│   ├── mul.mlir                    # FileCheck regression tests for scalar/tensor mul lowering
│   ├── canonicalize_add.mlir       # FileCheck tests for AddZero (x + 0 -> x) and AddSelf (x + x -> x * 2)
│   └── canonicalize_mul.mlir       # FileCheck tests for MulOne (x * 1 -> x) and MulZero (x * 0 -> 0)
└── tools/
    ├── arithmetic-opt.cpp          # Standalone MLIR optimizer driver
    └── CMakeLists.txt              # arithmetic-opt executable target
```

---

## Active Operations & Canonicalization Patterns

The active build focuses on **`addop`** and **`mulop`** over scalar and tensor `i32` and `f32` types (`AnyTypeOf<[I32, I32Tensor, F32, F32Tensor]>`), with `hasCanonicalizer = 1` enabled in `include/arithmeticops/arithmeticops.td`:

| Operation | Mnemonic | Traits | Canonicalization Patterns (`--canonicalize`) | Lowering (`--convert-arithmeticops-to-arith`) |
| :--- | :--- | :--- | :--- | :--- |
| **`addop`** | `arithmeticops.add` | `[SameOperandsAndResultType]` | • **`AddZero`**: `x + 0 -> x`, `0 + x -> x`<br>• **`AddSelf`**: `x + x -> arithmeticops.mul(x, 2)` | Lowered to `arith.addi` (integers) or `arith.addf` (floats) |
| **`mulop`** | `arithmeticops.mul` | `[SameOperandsAndResultType]` | • **`MulOne`**: `x * 1 -> x`, `1 * x -> x`<br>• **`MulZero`**: `x * 0 -> 0`, `0 * x -> 0` | Lowered to `arith.muli` (integers) or `arith.mulf` (floats) |

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

Automated canonicalization and lowering tests live in `lit_tests/` and use `llvm-lit` + `FileCheck` to verify IR transformations.

### 1. Build and Run the Full `lit` Test Suite via CMake

From inside `build/` (automatically recompiles `arithmetic-opt` if C++ files changed):

```bash
cmake --build . --target check-arithmetic
```

### 2. Run `llvm-lit` Directly (Fast Iteration on `.mlir` Tests)

Once `build/lit_tests/lit.site.cfg.py` has been generated by CMake, you can invoke `llvm-lit` directly from `build/` without triggering a CMake rebuild:

```bash
# Run all tests in lit_tests/ (lowering + canonicalization) with verbose output
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/

# Run individual canonicalization or lowering test files
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/canonicalize_add.mlir
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/canonicalize_mul.mlir
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/add.mlir
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/mul.mlir
```

### 3. Run an Individual `FileCheck` Pipeline Manually

```bash
./bin/arithmetic-opt ../lit_tests/canonicalize_add.mlir --canonicalize \
  | /home/mcw/mlir/llvm-project/build/bin/FileCheck ../lit_tests/canonicalize_add.mlir
```

---

## Running Pass Pipelines on the CLI

### 1. Run Canonicalization (`--canonicalize`)

Simplify algebraic identities (`x + 0`, `x + x -> x * 2`, `x * 1`, `x * 0`) while remaining inside `arithmeticops`:

```bash
./build/bin/arithmetic-opt lit_tests/canonicalize_add.mlir --canonicalize
./build/bin/arithmetic-opt lit_tests/canonicalize_mul.mlir --canonicalize
```

### 2. Canonicalize and Lower `arithmeticops` to Standard `arith` Dialect

Run `--canonicalize` first to eliminate redundant operations, then convert remaining `arithmeticops.add` and `arithmeticops.mul` operations into `arith.addi` / `arith.addf` and `arith.muli` / `arith.mulf`:

```bash
./build/bin/arithmetic-opt lit_tests/canonicalize_add.mlir \
  --canonicalize \
  --convert-arithmeticops-to-arith
```

### 3. Chain Built-In MLIR Passes to Lower All the Way to LLVM Dialect

Because `arithmetic-opt` registers MLIR's standard passes, you can chain the output directly into upstream LLVM lowering passes on the CLI:

```bash
./build/bin/arithmetic-opt lit_tests/mul.mlir \
  --canonicalize \
  --convert-arithmeticops-to-arith \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts
```