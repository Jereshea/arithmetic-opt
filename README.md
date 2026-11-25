# arithmetic-opt (`arithmeticops` MLIR Sample Dialect & Example Repo)

AAn end-to-end **MLIR example repository** and custom out-of-tree **MLIR sample dialect** (`arithmeticops`) with a standalone optimizer driver (`arithmetic-opt`). Built from scratch to explore modern MLIR compiler development, it demonstrates:

- **MLIR Operation Definition Specification (ODS):** Custom TableGen dialect and operation definitions (`add`, `mul`, `addcmul`, `fusedaddmul`).
- **Operator Fusion Pass (`--fuse-arithmetic-ops`):** Pattern-based graph fusion combining `arithmeticops.mul` + `arithmeticops.add` into a single `arithmeticops.fusedaddmul` operation.
- **Folding & Canonicalization (`--canonicalize`):** Hybrid in-place constant/identity folding (`hasFolder = 1`) and structural `OpRewritePattern` canonicalization (`hasCanonicalizer = 1`).
- **Progressive Dialect Lowering (`--convert-arithmeticops-to-arith`):** Dialect conversion to standard `arith` and `tensor` dialects, plus end-to-end lowering to LLVM IR (`mlir-runner`).
- **Regression Testing:** Automated `llvm-lit` + `FileCheck` test suite.

> New to MLIR? See **[GUIDE.md](GUIDE.md)** for a step-by-step walkthrough of TableGen (`mlir-tblgen`), C++ dialect/op registration, folding, canonicalization, operator fusion, lowering passes, and CMake setup.

---

## Project Structure

```text
arithmetic-opt/
├── CMakeLists.txt                  # Top-level CMake configuration
├── GUIDE.md                        # Step-by-step guide for building, fusing, canonicalizing, and lowering a custom MLIR dialect
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
│       ├── arithmeticops.td        # Active TableGen dialect & operation definitions (addop, mulop, addcmulop, fusedaddmulop)
│       ├── ArithmeticOpsFusion.h   # Operator fusion pattern population declaration
│       ├── ArithmeticOpsLowering.h # Conversion pattern population declaration
│       ├── Passes.h                # Pass declarations & registration hooks
│       ├── Passes.td               # TableGen pass definitions (--fuse-arithmetic-ops, --convert-arithmeticops-to-arith)
│       └── CMakeLists.txt          # mlir-tblgen code generation targets
├── lib/
│   └── arithmeticops/
│       ├── arithmetic.cpp          # Dialect initialization & registration (with Arith/Tensor dependentDialects)
│       ├── arithmeticop.cpp        # Operation definitions, fold() methods, and OpRewritePatterns
│       ├── ArithmeticOpsFusion.cpp # OpRewritePatterns fusing mul + add -> fusedaddmul
│       ├── ArithmeticOpsLowering.cpp # OpConversionPatterns (add, mul, addcmul, fusedaddmul -> arith & tensor dialects)
│       ├── Passes.cpp              # FuseArithmeticOpsPass & ConvertArithmeticOpsToArithPass implementations
│       └── CMakeLists.txt          # MLIRArithmeticOps static library target
├── lit_tests/
│   ├── CMakeLists.txt              # Configures lit.site.cfg.py and check-arithmetic target
│   ├── lit.cfg.py                  # llvm-lit suite configuration (ShTest & tool substitutions)
│   ├── lit.site.cfg.py.in          # CMake template for build-time paths
│   ├── add.mlir                    # FileCheck regression tests for scalar/tensor add lowering
│   ├── mul.mlir                    # FileCheck regression tests for scalar/tensor mul lowering
│   ├── addcmul.mlir                # FileCheck regression tests for tensor addcmul lowering
│   ├── fuse_addmul.mlir            # FileCheck regression tests for mul + add -> fusedaddmul fusion & lowering
│   ├── canonicalize_add.mlir       # FileCheck tests for addop folding (x + 0) & AddSelf (x + x -> x * 2)
│   ├── canonicalize_mul.mlir       # FileCheck tests for mulop folding (x * 1 -> x, x * 0 -> 0)
│   └── canonicalize_addcmul.mlir   # FileCheck tests for addcmulop folding & structural canonicalization
└── tools/
    ├── arithmetic-opt.cpp          # Standalone MLIR optimizer driver
    └── CMakeLists.txt              # arithmetic-opt executable target
```

---

## Active Operations, Fusion, Folding & Canonicalization

The active build defines **`addop`**, **`mulop`**, **`addcmulop`**, and **`fusedaddmulop`**, with `dependentDialects = ["mlir::arith::ArithDialect", "mlir::tensor::TensorDialect"]` declared on the dialect so `--canonicalize` and lowering passes can materialize `arith` and `tensor` operations seamlessly.

Canonicalization uses a **hybrid `fold()` + `OpRewritePattern` architecture**:
* **`fold()` (`hasFolder = 1`):** Handles in-place identity and zero simplifications that return an existing SSA operand without creating new operations.
* **`OpRewritePattern` (`hasCanonicalizer = 1`):** Handles structural transformations that replace an operation with newly constructed operations (e.g., `arith.constant`, `tensor.splat`, `arithmeticops.mul`, `arithmeticops.add`).

| Operation | Mnemonic | Traits & Types | `fold()` / Fusion Rules | `OpRewritePattern` Rules (`--canonicalize`) | Lowering (`--convert-arithmeticops-to-arith`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`addop`** | `arithmeticops.add` | `[Pure, SameOperandsAndResultType]`<br>`I32, F32, I32Tensor, F32Tensor` | • **Fold:** `x + 0 -> x`, `0 + x -> x`<br>• **Fuse (`--fuse-arithmetic-ops`):** `add(mul(a, b), c)` or `add(c, mul(a, b))` -> `fusedaddmul(a, b, c)` | • **`AddSelf`**: `x + x -> arithmeticops.mul(x, 2)` | Lowered to `arith.addi` (integers) or `arith.addf` (floats) |
| **`mulop`** | `arithmeticops.mul` | `[Pure, SameOperandsAndResultType]`<br>`I32, F32, I32Tensor, F32Tensor` | • **Fold:** `x * 1 -> x`, `1 * x -> x`, `x * 0 -> 0`, `0 * x -> 0`<br>• **Fuse (`--fuse-arithmetic-ops`):** Fused into `fusedaddmul` when feeding `add` | *(Handled via `fold()`)* | Lowered to `arith.muli` (integers) or `arith.mulf` (floats) |
| **`fusedaddmulop`** | `arithmeticops.fusedaddmul`<br>`(a * b) + c` | `[Pure, SameOperandsAndResultType]`<br>`a, b, c`: `I32, F32, I32Tensor, F32Tensor` | • Produced by `--fuse-arithmetic-ops` from `mul` + `add` chains (when `mul` has a single `add` user) | *(Fused target op)* | Lowered to `arith.muli`/`mulf` + `arith.addi`/`addf` |
| **`addcmulop`** | `arithmeticops.addcmul`<br>`input + value * (tensor1 * tensor2)` | `[Pure, AllShapesMatch, AllElementTypesMatch]`<br>`input, tensor1, tensor2`: `I32Tensor, F32Tensor`<br>`value`: `I32, F32` | • **Fold:** `input + 0 * (t1 * t2) -> input`<br>• **Fold:** `input + val * (dense<0> * t2) -> input`<br>• **Fold:** `input + val * (t1 * dense<0>) -> input` | • **`Input_Zero`**: `dense<0> + val * (t1 * t2) -> splat(val) * (t1 * t2)`<br>• **`Value_One`**: `input + 1 * (t1 * t2) -> input + (t1 * t2)`<br>• **`Tensor1_One`**: `input + val * (dense<1> * t2) -> input + (splat(val) * t2)`<br>• **`Tensor2_One`**: `input + val * (t1 * dense<1>) -> input + (splat(val) * t1)` | Lowered to `tensor.splat` + `arith.muli`/`mulf` + `arith.addi`/`addf` |

> **Archived Prototypes (`archive/`):** Earlier experimental ODS operations (`add_broadcasting_op`, `mul_broadcasting_op`, and `WeightedAddOp`) and their parse-only `.mlir` files are preserved in `archive/` outside the active CMake build tree.

---

## Prerequisites

* **LLVM / MLIR** built from source or installed (with `MLIRConfig.cmake`, `LLVMConfig.cmake`, `llvm-lit`, `FileCheck`, and `mlir-runner`)
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

Automated fusion, canonicalization, and lowering tests live in `lit_tests/` and use `llvm-lit` + `FileCheck` to verify IR transformations across all test files.

### 1. Build and Run the Full `lit` Test Suite via CMake

From inside `build/` (automatically recompiles `arithmetic-opt` if C++ files changed):

```bash
cmake --build . --target check-arithmetic
```

### 2. Run `llvm-lit` Directly (Fast Iteration on `.mlir` Tests)

Once `build/lit_tests/lit.site.cfg.py` has been generated by CMake, you can invoke `llvm-lit` directly from `build/` without triggering a CMake rebuild:

```bash
# Run all tests in lit_tests/ (fusion + lowering + canonicalization) with verbose output
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/

# Run individual fusion, canonicalization, or lowering test files
/home/mcw/mlir/llvm-project/build/bin/llvm-lit -v lit_tests/fuse_addmul.mlir
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

### 1. Fuse `mul` + `add` Operations (`--fuse-arithmetic-ops`)

Combine eligible `arithmeticops.mul` and `arithmeticops.add` pairs into `arithmeticops.fusedaddmul`:

```bash
./build/bin/arithmetic-opt lit_tests/fuse_addmul.mlir --fuse-arithmetic-ops
```

### 2. Run Canonicalization (`--canonicalize`)

Simplify algebraic identities and fold constant zero/one operands across `add`, `mul`, and `addcmul`:

```bash
./build/bin/arithmetic-opt lit_tests/canonicalize_add.mlir --canonicalize
./build/bin/arithmetic-opt lit_tests/canonicalize_mul.mlir --canonicalize
./build/bin/arithmetic-opt lit_tests/canonicalize_addcmul.mlir --canonicalize
```

### 3. Fuse, Canonicalize, and Lower `arithmeticops` to Standard `arith` & `tensor` Dialects

Run `--fuse-arithmetic-ops` and `--canonicalize` first to simplify and fuse operations, then convert remaining `arithmeticops` operations into `arith` and `tensor` dialect ops:

```bash
./build/bin/arithmetic-opt lit_tests/fuse_addmul.mlir \
  --fuse-arithmetic-ops \
  --canonicalize \
  --convert-arithmeticops-to-arith
```

### 4. End-to-End Lowering to LLVM IR & JIT Execution (`mlir-runner`)

Chain `arithmetic-opt` with upstream `mlir-opt` and `mlir-runner` to execute fused scalar IR directly on the CPU:

```bash
./build/bin/arithmetic-opt lit_tests/fuse_addmul.mlir \
  --fuse-arithmetic-ops \
  --convert-arithmeticops-to-arith \
  | /home/mcw/mlir/llvm-project/build/bin/mlir-opt \
      --convert-arith-to-llvm \
      --convert-func-to-llvm \
      --reconcile-unrealized-casts \
  | /home/mcw/mlir/llvm-project/build/bin/mlir-runner \
      -e main \
      -entry-point-result=i32 \
      --shared-libs=/home/mcw/mlir/llvm-project/build/lib/libmlir_runner_utils.so,/home/mcw/mlir/llvm-project/build/lib/libmlir_c_runner_utils.so
```