func.func @test_mul() -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arithmeticops.mul %c2 * %c3 : i32
  return %0 : i32
}