configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ARITHMETIC_TEST_DEPENDS
  arithmetic-opt
)

add_lit_testsuite(check-arithmetic "Running the arithmetic-opt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ARITHMETIC_TEST_DEPENDS}
)
set_target_properties(check-arithmetic PROPERTIES FOLDER "Tests")