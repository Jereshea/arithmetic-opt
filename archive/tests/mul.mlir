func.func @test_mul_scalar_int(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arithmeticops.mul %arg0 * %arg1 : i32
  return %0 : i32
}

func.func @test_mul_tensor_int(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  %0 = arithmeticops.mul %arg0 * %arg1 : tensor<?xi32>
  return %0 : tensor<?xi32>
}

func.func @test_mul_scalar_float(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arithmeticops.mul %arg0 * %arg1 : f32
  return %0 : f32
}

func.func @test_mul_tensor_float(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = arithmeticops.mul %arg0 * %arg1 : tensor<?xf32>
  return %0 : tensor<?xf32>
}