add_mlir_dialect(arithmeticops arithmeticops)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ArithmeticOps)
add_public_tablegen_target(MLIRArithmeticOpsPassesIncGen)