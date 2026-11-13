add_mlir_dialect_library(MLIRArithmeticOps
        arithmetic.cpp
        arithmeticop.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/arithmeticops

        DEPENDS
        MLIRarithmeticopsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )