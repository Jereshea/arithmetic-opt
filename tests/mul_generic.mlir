func.func @test_mul(%arg0: i32, %arg1: i32) -> i32 {
  %0 = "arithmeticops.mul"(%arg0, %arg1) : (i32, i32) -> i32
  return %0 : i32
}