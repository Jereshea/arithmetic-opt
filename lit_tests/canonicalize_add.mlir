// RUN: arithmetic-opt %s --canonicalize | FileCheck %s

/// Scalar i32: x + 0 -> x
// CHECK-LABEL: func.func @test_add_zero_scalar_int
// CHECK-SAME: (%[[ARG0:.*]]: i32)
// CHECK-NOT: arithmeticops.add
// CHECK: return %[[ARG0]] : i32
func.func @test_add_zero_scalar_int(%val_1: i32) -> i32 {
  %val_2 = arith.constant 0 : i32
  %result = "arithmeticops.add"(%val_1, %val_2) : (i32, i32) -> i32
  return %result : i32
}

/// Scalar f32: 0.0 + x -> x
// CHECK-LABEL: func.func @test_add_zero_scalar_float
// CHECK-SAME: (%[[ARG0:.*]]: f32)
// CHECK-NOT: arithmeticops.add
// CHECK: return %[[ARG0]] : f32
func.func @test_add_zero_scalar_float(%val_2: f32) -> f32 {
  %val_1 = arith.constant 0.0 : f32
  %result = "arithmeticops.add"(%val_1, %val_2) : (f32, f32) -> f32
  return %result : f32
}

/// Tensor i32: x + dense<0> -> x
// CHECK-LABEL: func.func @test_add_zero_tensor_int
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xi32>)
// CHECK-NOT: arithmeticops.add
// CHECK: return %[[ARG0]] : tensor<4xi32>
func.func @test_add_zero_tensor_int(%val_1: tensor<4xi32>) -> tensor<4xi32> {
  %val_2 = arith.constant dense<0> : tensor<4xi32>
  %result = "arithmeticops.add"(%val_1, %val_2) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

/// Tensor f32: dense<0.0> + x -> x
// CHECK-LABEL: func.func @test_add_zero_tensor_float
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>)
// CHECK-NOT: arithmeticops.add
// CHECK: return %[[ARG0]] : tensor<4xf32>
func.func @test_add_zero_tensor_float(%val_2: tensor<4xf32>) -> tensor<4xf32> {
  %val_1 = arith.constant dense<0.0> : tensor<4xf32>
  %result = "arithmeticops.add"(%val_1, %val_2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

/// Scalar i32: x + x -> x * 2
// CHECK-LABEL: func.func @test_add_self_scalar_int
// CHECK-SAME: (%[[ARG0:.*]]: i32)
// CHECK: %[[C2:.*]] = arith.constant 2 : i32
// CHECK-NOT: arithmeticops.add
// CHECK: %[[RES:.*]] = arithmeticops.mul %[[ARG0]] * %[[C2]] : i32
// CHECK: return %[[RES]] : i32
func.func @test_add_self_scalar_int(%val_1: i32) -> i32 {
  %result = "arithmeticops.add"(%val_1, %val_1) : (i32, i32) -> i32
  return %result : i32
}

/// Scalar f32: x + x -> x * 2.0
// CHECK-LABEL: func.func @test_add_self_scalar_float
// CHECK-SAME: (%[[ARG0:.*]]: f32)
// CHECK: %[[C2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-NOT: arithmeticops.add
// CHECK: %[[RES:.*]] = arithmeticops.mul %[[ARG0]] * %[[C2]] : f32
// CHECK: return %[[RES]] : f32
func.func @test_add_self_scalar_float(%val_1: f32) -> f32 {
  %result = "arithmeticops.add"(%val_1, %val_1) : (f32, f32) -> f32
  return %result : f32
}

/// Tensor i32: x + x -> x * dense<2>
// CHECK-LABEL: func.func @test_add_self_tensor_int
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xi32>)
// CHECK: %[[C2:.*]] = arith.constant dense<2> : tensor<4xi32>
// CHECK-NOT: arithmeticops.add
// CHECK: %[[RES:.*]] = arithmeticops.mul %[[ARG0]] * %[[C2]] : tensor<4xi32>
// CHECK: return %[[RES]] : tensor<4xi32>
func.func @test_add_self_tensor_int(%val_1: tensor<4xi32>) -> tensor<4xi32> {
  %result = "arithmeticops.add"(%val_1, %val_1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

/// Tensor f32: x + x -> x * dense<2.0>
// CHECK-LABEL: func.func @test_add_self_tensor_float
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>)
// CHECK: %[[C2:.*]] = arith.constant dense<2.000000e+00> : tensor<4xf32>
// CHECK-NOT: arithmeticops.add
// CHECK: %[[RES:.*]] = arithmeticops.mul %[[ARG0]] * %[[C2]] : tensor<4xf32>
// CHECK: return %[[RES]] : tensor<4xf32>
func.func @test_add_self_tensor_float(%val_1: tensor<4xf32>) -> tensor<4xf32> {
  %result = "arithmeticops.add"(%val_1, %val_1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}