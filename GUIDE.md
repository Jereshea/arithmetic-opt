# Step-by-Step Guide to Writing a Custom Out-of-Tree MLIR Dialect

This guide walks through building a custom MLIR dialect from scratch—explaining **why** each TableGen (`.td`), C++ header (`.h`), C++ source (`.cpp`), and `CMakeLists.txt` file exists and how `mlir-tblgen` connects them together.

---

## High-Level Workflow

1. **Define the Dialect & Operations** in TableGen (`include/<dialect>/<dialect>.td`).
2. **Inspect Generated Declarations** using `mlir-tblgen` to discover which MLIR C++ headers must be included in your `.h` files.
3. **Write C++ Headers (`.h`)** that include the required MLIR headers and pull in the generated `.h.inc` files.
4. **Write C++ Implementation Files (`.cpp`)** to implement `initialize()`, register operations via `addOperations<...>()`, and pull in the generated `.cpp.inc` definitions.
5. **Create the Optimizer Driver (`*-opt.cpp`)** to register your dialect and run `.mlir` test files.
6. **Configure `CMakeLists.txt` Files** so code generation runs before C++ compilation, then build with `ninja`.

---

## Step 1: Define the Dialect and Operations (`.td`)

Create your TableGen file (for example, `include/arithmeticops/arithmeticops.td`) to declare your dialect namespace and operations:

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

* Look at the generated class declaration (e.g., `class arithmeticopsDialect : public ::mlir::Dialect`) and any methods without `{ ... }` bodies (such as `void initialize();`) that you must implement in your `.cpp` file.
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

> **Common Gotcha (`BytecodeOpInterface`):** When an operation uses attributes (like `F32ArrayAttr`) or `Optional<...>` operands, `mlir-tblgen` automatically adds `::mlir::BytecodeOpInterface::Trait` to the op's base classes. Make sure to include `"mlir/Bytecode/BytecodeOpInterface.h"` in your op header (`include/arithmeticops/arithmeticop.h`) before `#include "arithmeticops/arithmeticops.h.inc"`.

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

The dialect `.cpp` file pulls in the generated dialect definitions (`arithmeticopsDialect.cpp.inc`), implements `initialize()`, and registers all custom operations using `addOperations<...>()`:

* **`GET_OP_LIST`:** Expands to a comma-separated list of all generated operation class names (e.g., `::arithmetic::addop, ::arithmetic::WeightedAddOp`) to pass directly into `addOperations<...>()`.

```cpp
#include "arithmeticops/arithmetic.h"
#include "arithmeticops/arithmeticop.h"

using namespace mlir;
using namespace arithmetic;

#include "arithmeticops/arithmeticopsDialect.cpp.inc"

void arithmeticopsDialect::initialize() {
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

## Step 4: Create the Driver Tool (`tools/arithmetic-opt.cpp`)

To parse, verify, and print `.mlir` files using your dialect, create a standalone `mlir-opt`-style driver using `llvm-project/mlir/examples/standalone/standalone-opt/standalone-opt.cpp` as a template:

1. Create a `DialectRegistry`.
2. Register standard dialects you need in tests (such as `func::FuncDialect` and `arith::ArithDialect`) along with your custom dialect (`registry.insert<arithmetic::arithmeticopsDialect>()`).
3. Pass the registry to `mlir::MlirOptMain(argc, argv, "Arithmetic optimizer driver\n", registry)`.

---

## Step 5: Configure `CMakeLists.txt` Files

### Why Build Order Matters

Building an MLIR dialect happens in **three strict stages**:
1. **Stage 1 (`include/`):** `mlir-tblgen` generates `.h.inc` and `.cpp.inc` files from your `.td` file.
2. **Stage 2 (`lib/`):** The C++ compiler compiles `arithmetic.cpp` and `arithmeticop.cpp` into a dialect library.
3. **Stage 3 (`tools/`):** The linker builds the `arithmetic-opt` executable against your dialect library and MLIR libraries.

If CMake tries to compile `arithmetic.cpp` (Stage 2) before `mlir-tblgen` finishes generating `arithmeticops.cpp.inc` (Stage 1), the build fails immediately with `No such file or directory`. Splitting `CMakeLists.txt` by subdirectory and declaring explicit `DEPENDS` (such as `MLIRArithmeticOpsIncGen`) guarantees Stage 1 $\rightarrow$ Stage 2 $\rightarrow$ Stage 3 always runs in the exact right sequence.

### Upstream Standalone Reference Files

You can inspect the official MLIR standalone template directly in your local LLVM checkout or on GitHub:

* **Root `CMakeLists.txt`:** [CMakeLists.txt](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/CMakeLists.txt)
* **TableGen `CMakeLists.txt`:** [include/Standalone/CMakeLists.txt](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/include/Standalone/CMakeLists.txt)
* **Dialect Library `CMakeLists.txt`:** [lib/Standalone/CMakeLists.txt](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/lib/Standalone/CMakeLists.txt)
* **Driver Tool `CMakeLists.txt`:** [standalone-opt/CMakeLists.txt](https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/standalone-opt/CMakeLists.txt)

---

## Step 6: Configure and Build with CMake & Ninja

From your project root directory, create a `build/` folder, point CMake to your built/installed `MLIR_DIR` and `LLVM_DIR`, and compile:

```bash
cd /path/to/your/custom_dialect
mkdir -p build && cd build

cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm

ninja arithmetic-opt
```

*(Note: If LLVM/MLIR was installed system-wide via `ninja install`, use `/usr/local/lib/cmake/mlir` and `/usr/local/lib/cmake/llvm`.)*

### Test Your Dialect

Once `ninja arithmetic-opt` succeeds, run your driver on a `.mlir` test file:

```bash
./bin/arithmetic-opt ../test/your_test.mlir
```