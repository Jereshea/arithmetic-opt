func.func @test_mul_broadcast_3d() -> tensor<2x2x2xi32> {
  %c2 = arith.constant 2 : i32
  %t1 = arith.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi32>
  %0 = arithmeticops.mul_broadcasting_op %c2 * %t1 : i32, tensor<2x2x2xi32> -> tensor<2x2x2xi32>
  return %0 : tensor<2x2x2xi32>
}