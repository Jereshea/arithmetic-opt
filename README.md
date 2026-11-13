# arithmetic-opt (`arithmeticops` MLIR Dialect)

A custom out-of-tree MLIR dialect (`arithmeticops`) and standalone optimizer driver (`arithmetic-opt`) implementing element-wise and broadcasting arithmetic operations over `i32` scalars and `i32` tensors.

---

## Project Structure

```text
arithmetic-opt/
├── CMakeLists.txt                  # Top-level CMake configuration
├── include/
│   └── arithmeticops/
│       ├── arithmetic.h            # Dialect declaration
│       ├── arithmeticop.h          # Operation class declarations
│       ├── arithmeticops.td        # TableGen dialect & operation definitions
│       └── CMakeLists.txt          # mlir-tblgen code generation targets
├── lib/
│   └── arithmeticops/
│       ├── arithmetic.cpp          # Dialect initialization & registration
│       ├── arithmeticop.cpp        # Generated operation definitions
│       └── CMakeLists.txt          # MLIRArithmeticOps static library target
├── tools/
│   ├── arithmetic-opt.cpp          # Standalone MLIR optimizer driver
│   └── CMakeLists.txt              # arithmetic-opt executable target
└── test/
    ├── add.mlir                    # Scalar element-wise addition test
    ├── add_broadcasting.mlir       # Scalar-to-tensor addition broadcast test
    ├── mul.mlir                    # Constant scalar multiplication test
    ├── mul_generic.mlir            # Generic syntax multiplication round-trip test
    └── mul_broadcasting.mlir       # Scalar-to-3D-tensor multiplication broadcast test
```

---

## Supported Operations

| Operation | Mnemonic | Traits / Behavior | Assembly Format |
| :--- | :--- | :--- | :--- |
| **`addop`** | `arithmeticops.add` | `[SameOperandsAndResultType]` — enforces identical `i32` or `tensor<...xi32>` types across operands and result | Generic quoted syntax |
| **`add_broadcasting_op`** | `arithmeticops.add_broadcasting_op` | Allows mixed `i32` scalar and `tensor<...xi32>` operands for broadcasted addition | Generic quoted syntax |
| **`mulop`** | `arithmeticops.mul` | `[SameOperandsAndResultType]` — enforces identical `i32` or `tensor<...xi32>` types across operands and result | `%lhs * %rhs : type` |
| **`mulbroadcasting_op`** | `arithmeticops.mul_broadcasting_op` | Allows mixed `i32` scalar and `tensor<...xi32>` operands for broadcasted multiplication | `%lhs * %rhs : lhs_type, rhs_type -> res_type` |

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

## Running Tests

Run `arithmetic-opt` on an individual `.mlir` test file:

```bash
./build/bin/arithmetic-opt test/mul_broadcasting.mlir
```

Run the entire test suite in `test/`:

```bash
for f in test/*.mlir; do
  echo "=== Testing $f ==="
  ./build/bin/arithmetic-opt "$f"
done
```

### Example IR (`test/mul_broadcasting.mlir`)

```mlir
func.func @test_mul_broadcast_3d() -> tensor<2x2x2xi32> {
  %c2_i32 = arith.constant 2 : i32
  %cst = arith.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi32>
  %0 = arithmeticops.mul_broadcasting_op %c2_i32 * %cst : i32, tensor<2x2x2xi32> -> tensor<2x2x2xi32>
  return %0 : tensor<2x2x2xi32>
}
```
