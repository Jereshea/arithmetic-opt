// RUN: arithmetic-opt %s --canonicalize | FileCheck %s

/// Scalar i32: x * 1 -> x
// CHECK-LABEL: func.func @test_mul_one_scalar_int
// CHECK-SAME: (%[[ARG0:.*]]: i32)
// CHECK-NOT: arithmeticops.mul
// CHECK: return %[[ARG0]] : i32
func.func @test_mul_one_scalar_int(%val_1: i32) -> i32 {
  %val_2 = arith.constant 1 : i32
  %result = "arithmeticops.mul"(%val_1, %val_2) : (i32, i32) -> i32
  return %result : i32
}

/// Scalar f32: 1.0 * x -> x
// CHECK-LABEL: func.func @test_mul_one_scalar_float
// CHECK-SAME: (%[[ARG0:.*]]: f32)
// CHECK-NOT: arithmeticops.mul
// CHECK: return %[[ARG0]] : f32
func.func @test_mul_one_scalar_float(%val_2: f32) -> f32 {
  %val_1 = arith.constant 1.0 : f32
  %result = "arithmeticops.mul"(%val_1, %val_2) : (f32, f32) -> f32
  return %result : f32
}

/// Tensor i32: x * dense<1> -> x
// CHECK-LABEL: func.func @test_mul_one_tensor_int
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xi32>)
// CHECK-NOT: arithmeticops.mul
// CHECK: return %[[ARG0]] : tensor<4xi32>
func.func @test_mul_one_tensor_int(%val_1: tensor<4xi32>) -> tensor<4xi32> {
  %val_2 = arith.constant dense<1> : tensor<4xi32>
  %result = "arithmeticops.mul"(%val_1, %val_2) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

/// Tensor f32: dense<1.0> * x -> x
// CHECK-LABEL: func.func @test_mul_one_tensor_float
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>)
// CHECK-NOT: arithmeticops.mul
// CHECK: return %[[ARG0]] : tensor<4xf32>
func.func @test_mul_one_tensor_float(%val_2: tensor<4xf32>) -> tensor<4xf32> {
  %val_1 = arith.constant dense<1.0> : tensor<4xf32>
  %result = "arithmeticops.mul"(%val_1, %val_2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

/// Scalar i32: x * 0 -> 0
// CHECK-LABEL: func.func @test_mul_zero_scalar_int
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NOT: arithmeticops.mul
// CHECK: return %[[C0]] : i32
func.func @test_mul_zero_scalar_int(%val_1: i32) -> i32 {
  %val_2 = arith.constant 0 : i32
  %result = "arithmeticops.mul"(%val_1, %val_2) : (i32, i32) -> i32
  return %result : i32
}

/// Scalar f32: 0.0 * x -> 0.0
// CHECK-LABEL: func.func @test_mul_zero_scalar_float
// CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NOT: arithmeticops.mul
// CHECK: return %[[C0]] : f32
func.func @test_mul_zero_scalar_float(%val_2: f32) -> f32 {
  %val_1 = arith.constant 0.0 : f32
  %result = "arithmeticops.mul"(%val_1, %val_2) : (f32, f32) -> f32
  return %result : f32
}

/// Tensor i32: x * dense<0> -> dense<0>
// CHECK-LABEL: func.func @test_mul_zero_tensor_int
// CHECK: %[[C0:.*]] = arith.constant dense<0> : tensor<4xi32>
// CHECK-NOT: arithmeticops.mul
// CHECK: return %[[C0]] : tensor<4xi32>
func.func @test_mul_zero_tensor_int(%val_1: tensor<4xi32>) -> tensor<4xi32> {
  %val_2 = arith.constant dense<0> : tensor<4xi32>
  %result = "arithmeticops.mul"(%val_1, %val_2) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

/// Tensor f32: dense<0.0> * x -> dense<0.000000e+00>
// CHECK-LABEL: func.func @test_mul_zero_tensor_float
// CHECK: %[[C0:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NOT: arithmeticops.mul
// CHECK: return %[[C0]] : tensor<4xf32>
func.func @test_mul_zero_tensor_float(%val_2: tensor<4xf32>) -> tensor<4xf32> {
  %val_1 = arith.constant dense<0.0> : tensor<4xf32>
  %result = "arithmeticops.mul"(%val_1, %val_2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}