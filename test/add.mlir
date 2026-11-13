func.func @test_add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = "arithmeticops.add"(%arg0, %arg1) : (i32, i32) -> i32
  return %0 : i32
}