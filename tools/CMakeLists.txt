set(LIBS
    MLIRArithmeticOps
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
  )
add_llvm_executable(arithmetic-opt arithmetic-opt.cpp)

llvm_update_compile_flags(arithmetic-opt)
target_link_libraries(arithmetic-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(arithmetic-opt)