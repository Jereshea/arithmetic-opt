# Step-by-Step Guide to Writing and Lowering a Custom Out-of-Tree MLIR Dialect

This guide walks through building a custom MLIR dialect from scratch and lowering it to standard MLIR (`arith`) and LLVM IR—explaining **why** each TableGen (`.td`), C++ header (`.h`), C++ source (`.cpp`), and `CMakeLists.txt` file exists and how `mlir-tblgen` connects them together.

---

## High-Level Workflow

1. **Define the Dialect & Operations** in TableGen (`include/<dialect>/<dialect>.td`).
2. **Inspect Generated Declarations** using `mlir-tblgen` to discover which MLIR C++ headers must be included in your `.h` files.
3. **Write C++ Headers (`.h`)** that include the required MLIR headers and pull in the generated `.h.inc` files.
4. **Write C++ Implementation Files (`.cpp`)** to implement `initialize()`, register operations via `addOperations<...>()`, and pull in the generated `.cpp.inc` definitions.
5. **Define and Implement Lowering Passes** (`Passes.td`, `Passes.h`, `ArithmeticOpsLowering.cpp`, and `Passes.cpp`) using `OpConversionPattern` and `ConversionTarget`.
6. **Create the Optimizer Driver (`*-opt.cpp`)** to register your dialect, lowering passes, and optional end-to-end pipelines.
7. **Configure `CMakeLists.txt` Files** so TableGen code generation runs before C++ compilation, then build with `ninja` and test.

---

# Part 1: Defining the Custom Dialect and Operations

## Step 1: Define the Dialect and Operations (`.td`)

Create your TableGen file (`include/arithmeticops/arithmeticops.td`) to declare your dialect namespace and operations:

* **Dialect definition:** Sets the MLIR prefix (e.g., `"arithmeticops"`) and C++ namespace (e.g., `"::arithmetic"`).
* **Operation definitions (`Op<...>`):** Specify `arguments` (operands + attributes), `results`, traits (e.g., `SameOperandsAndResultType`), and optional `assemblyFormat`.

---

## Step 2: Create C++ Header Files (`.h`)

Before writing your `.h` files manually, run `mlir-tblgen` to inspect the exact C++ classes and base traits it generates. Knowing the base classes tells you which MLIR headers you need to `#include` so C++ compilation succeeds.

### 2.1 Inspect Dialect Declarations

```bash
mlir-tblgen \
  -gen-dialect-decls \
  -I /path/to/llvm-project/mlir/include \
  include/arithmeticops/arithmeticops.td
```

* Look at the generated class declaration (e.g., `class arithmeticops : public ::mlir::Dialect`) and any methods without `{ ... }` bodies (such as `void initialize();`) that you must implement in your `.cpp` file.
* Because it inherits from `::mlir::Dialect`, your dialect header (`include/arithmeticops/arithmetic.h`) must include `"mlir/IR/Dialect.h"`.

### 2.2 Inspect Operation Declarations

To see a quick summary of generated macros and class inheritance lines:

```bash
mlir-tblgen \
  -gen-op-decls \
  -I /path/to/llvm-project/mlir/include \
  include/arithmeticops/arithmeticops.td \
| grep -E "^#ifdef|class .* : public|MLIR_DECLARE"
```

For the full generated header output:

```bash
mlir-tblgen \
  -gen-op-decls \
  -I /path/to/llvm-project/mlir/include \
  include/arithmeticops/arithmeticops.td
```

### 2.3 Find the Required MLIR Headers for Base Classes & Traits

For each generated op class, examine its `public ::mlir::Op<...>` inheritance list. For example:

```cpp
class addop : public ::mlir::Op<
    addop,
    ::mlir::OpTrait::ZeroRegions,
    ::mlir::OpTrait::OneResult,
    ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl,
    ::mlir::OpTrait::ZeroSuccessors,
    ::mlir::OpTrait::NOperands<2>::Impl,
    ::mlir::OpTrait::OpInvariants> {
```

If a base class or trait triggers an `incomplete type` or `not declared in this scope` compiler error, `grep` the MLIR include directory to find which header defines it:

```bash
grep -rn "class Op " /path/to/llvm-project/mlir/include/mlir/
# Output: .../mlir/include/mlir/IR/OpDefinition.h:1707:class Op : public OpState, public Traits<ConcreteType>...
```

> **Common Gotcha (`BytecodeOpInterface`):** When an operation uses attributes (like `F32ArrayAttr`) or `Optional<...>` operands, `mlir-tblgen` automatically adds `::mlir::BytecodeOpInterface::Trait` to the op's base classes. Include `"mlir/Bytecode/BytecodeOpInterface.h"` in your op header (`include/arithmeticops/arithmeticop.h`) before `#include "arithmeticops/arithmeticops.h.inc"`.

### Example Op Header (`include/arithmeticops/arithmeticop.h`)

```cpp
#ifndef ARITHMETICOP_H
#define ARITHMETICOP_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "arithmeticops/arithmeticops.h.inc"

#endif // ARITHMETICOP_H
```

---

## Step 3: Create C++ Implementation Files (`.cpp`)

In C++, every method declared inside a generated class (anything ending with `;`) must have a matching `{ ... }` function body compiled in a `.cpp` file, or linking will fail with `undefined reference` errors.

### 3.1 Inspect Generated `.cpp.inc` Definitions

* **Dialect definitions (`-gen-dialect-defs`):**
  ```bash
  mlir-tblgen \
    -gen-dialect-defs \
    -I /path/to/llvm-project/mlir/include \
    include/arithmeticops/arithmeticops.td \
  | grep -E "^#ifdef|MLIR_DEFINE|::initialize"
  ```
* **Operation definitions (`-gen-op-defs`):**
  ```bash
  mlir-tblgen \
    -gen-op-defs \
    -I /path/to/llvm-project/mlir/include \
    include/arithmeticops/arithmeticops.td \
  | grep -E "^#ifdef|MLIR_DEFINE"
  ```

### 3.2 Dialect Registration (`lib/arithmeticops/arithmetic.cpp`)

The dialect `.cpp` file pulls in the generated dialect definitions, implements `initialize()`, and registers all custom operations using `addOperations<...>()`:

* **`GET_OP_LIST`:** Expands to a comma-separated list of all generated operation class names (e.g., `::arithmetic::addop, ::arithmetic::mulop`) to pass directly into `addOperations<...>()`.

```cpp
#include "arithmeticops/arithmetic.h"
#include "arithmeticops/arithmeticop.h"

using namespace mlir;
using namespace arithmetic;

#include "arithmeticops/arithmeticopsDialect.cpp.inc"

void arithmeticops::initialize() {
  addOperations<
#define GET_OP_LIST
#include "arithmeticops/arithmeticops.cpp.inc"
      >();
}
```

### 3.3 Operation Implementations (`lib/arithmeticops/arithmeticop.cpp`)

The operation `.cpp` file compiles the full C++ method bodies (`build()`, `verifyInvariants()`, getters, parsers, printers, etc.) generated by `mlir-tblgen`:

* **`GET_OP_CLASSES`:** Expands to the complete C++ definitions of every operation class in your `.td` file.

```cpp
#include "arithmeticops/arithmeticop.h"
#include "arithmeticops/arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace arithmetic;

#define GET_OP_CLASSES
#include "arithmeticops/arithmeticops.cpp.inc"
```

---

# Part 2: Lowering Passes (`arithmeticops` $\rightarrow$ `arith` $\rightarrow$ LLVM)

Once your dialect parses and verifies cleanly, lower your custom operations into standard MLIR dialects (such as `arith`) so upstream MLIR passes can lower the IR all the way to LLVM IR and machine code.

## Step 4: Declare the Lowering Pass in TableGen (`include/arithmeticops/Passes.td`)

Create `include/arithmeticops/Passes.td` inheriting from `mlir/Pass/PassBase.td`:

```tablegen
#ifndef ARITHMETICOPS_PASSES
#define ARITHMETICOPS_PASSES

include "mlir/Pass/PassBase.td"

def ConvertArithmeticOpsToArith : Pass<"convert-arithmeticops-to-arith", "mlir::ModuleOp"> {
  let summary = "Lower arithmeticops dialect operations to standard Arith dialect";
  let description = [{
    Converts custom operations in the `arithmeticops` dialect (such as `add` and `mul`)
    into standard `arith` dialect operations.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif // ARITHMETICOPS_PASSES
```

### Key Fields Explained
* **`def ConvertArithmeticOpsToArith`:** Generates the CRTP base class `impl::ConvertArithmeticOpsToArithBase<Derived>` and the `GEN_PASS_DEF_CONVERTARITHMETICOPSTOARITH` macro.
* **`"convert-arithmeticops-to-arith"`:** The CLI flag registered in `arithmetic-opt`.
* **`"mlir::ModuleOp"`:** The root operation the pass runs on (`getOperation()` returns a `ModuleOp`).
* **`dependentDialects`:** Tells MLIR to load `mlir::arith::ArithDialect` automatically before creating `arith.addi` / `arith.muli`, even if the input `.mlir` file contained no `arith` operations initially.

### Inspect Generated Pass Declarations

```bash
mlir-tblgen \
  -gen-pass-decls \
  -name ArithmeticOps \
  -I /path/to/llvm-project/mlir/include \
  include/arithmeticops/Passes.td
```

> **How Registration Function Names Are Formed:** Passing `-name ArithmeticOps` to `mlir-tblgen -gen-pass-decls` generates `registerArithmeticOpsPasses()` inside whatever C++ namespace wraps `#define GEN_PASS_REGISTRATION` in `Passes.h`.

---

## Step 5: Create the Pass Header (`include/arithmeticops/Passes.h`)

Expose both the pass declarations and registration hooks inside `namespace arithmetic`:

```cpp
#ifndef ARITHMETICOPS_PASSES_H
#define ARITHMETICOPS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace arithmetic {

#define GEN_PASS_DECL
#include "arithmeticops/Passes.h.inc"

void populateArithmeticOpsToArithConversionPatterns(
    mlir::RewritePatternSet &patterns);

#define GEN_PASS_REGISTRATION
#include "arithmeticops/Passes.h.inc"

} // namespace arithmetic

#endif // ARITHMETICOPS_PASSES_H
```

---

## Step 6: Write Rewrite Patterns (`lib/arithmeticops/ArithmeticOpsLowering.cpp`)

Each custom operation gets an `OpConversionPattern<SourceOp>` that matches the custom op and replaces it with the corresponding `arith` operation via `ConversionPatternRewriter`:

```cpp
#include "arithmeticops/ArithmeticOpsLowering.h"
#include "arithmeticops/arithmeticop.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct AddOpConverter : public OpConversionPattern<arithmetic::addop> {
  using OpConversionPattern<arithmetic::addop>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arithmetic::addop op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getVal1(),
                                               adaptor.getVal2());
    return success();
  }
};

struct MulOpConverter : public OpConversionPattern<arithmetic::mulop> {
  using OpConversionPattern<arithmetic::mulop>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arithmetic::mulop op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::MulIOp>(op, adaptor.getVal1(),
                                               adaptor.getVal2());
    return success();
  }
};

} // namespace

void arithmetic::populateArithmeticOpsToArithConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AddOpConverter, MulOpConverter>(patterns.getContext());
}
```

* **`OpAdaptor adaptor`:** Provides type-safe getters (`adaptor.getVal1()`, `adaptor.getVal2()`) for converted operands.
* **`RewritePatternSet`:** Acts as a pattern toolbox that indexes each converter by its root operation name (`arithmeticops.add`, `arithmeticops.mul`) for $O(1)$ dispatch during traversal.

---

## Step 7: Implement the Conversion Pass (`lib/arithmeticops/Passes.cpp`)

Implement `runOnOperation()` by configuring `ConversionTarget`, populating `RewritePatternSet`, and invoking `applyPartialConversion`:

```cpp
#include "arithmeticops/Passes.h"
#include "arithmeticops/ArithmeticOpsLowering.h"
#include "arithmeticops/arithmetic.h"
#include "arithmeticops/arithmeticop.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace arithmetic {
#define GEN_PASS_DEF_CONVERTARITHMETICOPSTOARITH
#include "arithmeticops/Passes.h.inc"
} // namespace arithmetic

using namespace mlir;

namespace {

struct ConvertArithmeticOpsToArithPass
    : public arithmetic::impl::ConvertArithmeticOpsToArithBase<
          ConvertArithmeticOpsToArithPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();
    target.addIllegalDialect<arithmetic::arithmeticops>();

    RewritePatternSet patterns(context);
    arithmetic::populateArithmeticOpsToArithConversionPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
```

### Understanding `ConversionTarget` & Conversion Drivers
MLIR classifies every operation into **three states**:
1. **Legal (`addLegalDialect` / `addLegalOp`):** Allowed in the output IR and skipped by rewriters. **Mandatory** for target dialects created by your patterns (`arith::ArithDialect`), or MLIR will reject the replacement op.
2. **Unknown (Default):** Neither legal nor illegal. `applyPartialConversion` attempts to rewrite Unknown ops if a pattern matches, and leaves them untouched if no pattern matches.
3. **Illegal (`addIllegalDialect` / `addIllegalOp`):** Must be eliminated by a matching pattern. If any illegal op survives, `applyPartialConversion` fails and triggers `signalPassFailure()`.

* **`applyPartialConversion` vs. `applyFullConversion`:** Use `applyPartialConversion` when unrelated dialects (`func`, `scf`, `memref`) should pass through untouched; use `applyFullConversion` only when *every* operation in the module must land in explicitly legal dialects.
* **`std::move(patterns)`:** Transfers ownership of the non-copyable `RewritePatternSet` into a thread-safe `FrozenRewritePatternSet`.

---

# Part 3: Driver Tool, Build Configuration, and Testing

## Step 8: Create the Driver Tool (`tools/arithmetic-opt.cpp`)

Register standard MLIR dialects/passes, your custom dialect, your custom pass registration hook, and an optional end-to-end `PassPipelineRegistration` (`--arithmetic-to-llvm`):

```cpp
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "arithmeticops/Passes.h"
#include "arithmeticops/arithmetic.h"
#include "arithmeticops/arithmeticop.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // 1. Register standard MLIR dialects, passes, and custom passes
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  arithmetic::registerArithmeticOpsPasses();

  // 2. Register an end-to-end lowering pipeline flag: --arithmetic-to-llvm
  mlir::PassPipelineRegistration<>(
      "arithmetic-to-llvm",
      "Lower arithmeticops dialect all the way to the LLVM dialect",
      [](mlir::OpPassManager &pm) {
        pm.addPass(arithmetic::createConvertArithmeticOpsToArithPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      });

  // 3. Register your custom dialect
  registry.insert<arithmetic::arithmeticops>();

  // 4. Launch the standard mlir-opt CLI loop
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Arithmetic optimizer driver\n", registry));
}
```

---

## Step 9: Configure `CMakeLists.txt` Files

### Why Build Order Matters

Building an MLIR dialect and its passes happens in **three strict stages**:
1. **Stage 1 (`include/arithmeticops/CMakeLists.txt`):** `mlir-tblgen` generates `.h.inc` and `.cpp.inc` files from `arithmeticops.td` and `Passes.td` (e.g., `mlir_tablegen(Passes.h.inc -gen-pass-decls -name ArithmeticOps)`).
2. **Stage 2 (`lib/arithmeticops/CMakeLists.txt`):** Compiles `arithmetic.cpp`, `arithmeticop.cpp`, `ArithmeticOpsLowering.cpp`, and `Passes.cpp` into `MLIRArithmeticOps`, linking `MLIRIR`, `MLIRPass`, `MLIRTransforms`, `MLIRArithDialect`, and `MLIRFuncDialect`.
3. **Stage 3 (`tools/CMakeLists.txt`):** Links the `arithmetic-opt` executable against `MLIRArithmeticOps` and the upstream LLVM conversion libraries (`MLIRArithToLLVM`, `MLIRFuncToLLVM`, `MLIRReconcileUnrealizedCasts`).

Declaring explicit `DEPENDS` (such as `MLIRArithmeticOpsIncGen` and `MLIRArithmeticOpsPassesIncGen`) guarantees Stage 1 $\rightarrow$ Stage 2 $\rightarrow$ Stage 3 always runs in order.

### Upstream Reference Files

* **Root `CMakeLists.txt`:** [CMakeLists.txt](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/CMakeLists.txt)
* **TableGen `CMakeLists.txt`:** [include/Standalone/CMakeLists.txt](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/include/Standalone/CMakeLists.txt)
* **Dialect Library `CMakeLists.txt`:** [lib/Standalone/CMakeLists.txt](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/lib/Standalone/CMakeLists.txt)
* **Driver Tool `CMakeLists.txt`:** [standalone-opt/CMakeLists.txt](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/standalone-opt/CMakeLists.txt)
* **Conversion Pass Example:** `mlir/lib/Conversion/MathToArith/MathToArithPass.cpp`

---

## Step 10: Build and Verify Lowering

From your project root directory, configure with CMake and compile with `ninja`:

```bash
cd /path/to/arithmetic-opt
mkdir -p build && cd build

cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm

ninja arithmetic-opt
```

### Verify Parsing, Dialect Lowering, and LLVM IR Lowering

1. **Parse and verify custom dialect syntax:**
   ```bash
   ./bin/arithmetic-opt ../test/add.mlir
   ```

2. **Lower `arithmeticops` $\rightarrow$ `arith`:**
   ```bash
   ./bin/arithmetic-opt ../test/add.mlir --convert-arithmeticops-to-arith
   ```

3. **Lower end-to-end `arithmeticops` $\rightarrow$ `arith` $\rightarrow$ `llvm` dialect:**
   ```bash
   ./bin/arithmetic-opt ../test/mul.mlir --arithmetic-to-llvm
   ```