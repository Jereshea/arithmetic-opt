// Reference: https://github.com/llvm/llvm-project/blob/main/mlir/examples/standalone/standalone-opt/standalone-opt.cpp

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "arithmeticops/arithmetic.h"
#include "arithmeticops/arithmeticop.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // 1. Register standard MLIR dialects 
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  // 2. Register your custom dialect 
  registry.insert<arithmetic::arithmeticops>();

  // 3. Launch the standard mlir-opt CLI loop
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Arithmetic optimizer driver\n", registry));
}