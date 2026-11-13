func.func @test_add_broadcast(%arg0: i32, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "arithmeticops.add_broadcasting_op"(%arg0, %arg1) : (i32, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}