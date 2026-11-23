// RUN: arithmetic-opt %s --canonicalize | FileCheck %s

// 1. input + 0 * (tensor1 * tensor2) -> input (i32)
// CHECK-LABEL: func.func @test_fold_zero_value_int(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xi32>, %[[T1:.*]]: tensor<4xi32>, %[[T2:.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         return %[[IN]] : tensor<4xi32>
func.func @test_fold_zero_value_int(%in: tensor<4xi32>, %t1: tensor<4xi32>, %t2: tensor<4xi32>) -> tensor<4xi32> {
  %zero = arith.constant 0 : i32
  %0 = "arithmeticops.addcmul"(%in, %zero, %t1, %t2) : (tensor<4xi32>, i32, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// 2. input + 0.0 * (tensor1 * tensor2) -> input (f32)
// CHECK-LABEL: func.func @test_fold_zero_value_float(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>, %[[T1:.*]]: tensor<4xf32>, %[[T2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         return %[[IN]] : tensor<4xf32>
func.func @test_fold_zero_value_float(%in: tensor<4xf32>, %t1: tensor<4xf32>, %t2: tensor<4xf32>) -> tensor<4xf32> {
  %zero = arith.constant 0.0 : f32
  %0 = "arithmeticops.addcmul"(%in, %zero, %t1, %t2) : (tensor<4xf32>, f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// 3. input + value * (dense<0> * tensor2) -> input (i32)
// CHECK-LABEL: func.func @test_fold_zero_tensor1_int(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xi32>, %[[VAL:.*]]: i32, %[[T2:.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         return %[[IN]] : tensor<4xi32>
func.func @test_fold_zero_tensor1_int(%in: tensor<4xi32>, %val: i32, %t2: tensor<4xi32>) -> tensor<4xi32> {
  %zero_t = arith.constant dense<0> : tensor<4xi32>
  %0 = "arithmeticops.addcmul"(%in, %val, %zero_t, %t2) : (tensor<4xi32>, i32, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// 4. input + value * (dense<0.0> * tensor2) -> input (f32)
// CHECK-LABEL: func.func @test_fold_zero_tensor1_float(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>, %[[VAL:.*]]: f32, %[[T2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         return %[[IN]] : tensor<4xf32>
func.func @test_fold_zero_tensor1_float(%in: tensor<4xf32>, %val: f32, %t2: tensor<4xf32>) -> tensor<4xf32> {
  %zero_t = arith.constant dense<0.0> : tensor<4xf32>
  %0 = "arithmeticops.addcmul"(%in, %val, %zero_t, %t2) : (tensor<4xf32>, f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// 5. input + value * (tensor1 * dense<0>) -> input (i32)
// CHECK-LABEL: func.func @test_fold_zero_tensor2_int(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xi32>, %[[VAL:.*]]: i32, %[[T1:.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         return %[[IN]] : tensor<4xi32>
func.func @test_fold_zero_tensor2_int(%in: tensor<4xi32>, %val: i32, %t1: tensor<4xi32>) -> tensor<4xi32> {
  %zero_t = arith.constant dense<0> : tensor<4xi32>
  %0 = "arithmeticops.addcmul"(%in, %val, %t1, %zero_t) : (tensor<4xi32>, i32, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// 6. input + value * (tensor1 * dense<0.0>) -> input (f32)
// CHECK-LABEL: func.func @test_fold_zero_tensor2_float(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>, %[[VAL:.*]]: f32, %[[T1:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         return %[[IN]] : tensor<4xf32>
func.func @test_fold_zero_tensor2_float(%in: tensor<4xf32>, %val: f32, %t1: tensor<4xf32>) -> tensor<4xf32> {
  %zero_t = arith.constant dense<0.0> : tensor<4xf32>
  %0 = "arithmeticops.addcmul"(%in, %val, %t1, %zero_t) : (tensor<4xf32>, f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// 7. input + 2.0 * (tensor1 * tensor2) -> unchanged (non-zero/non-one)
// CHECK-LABEL: func.func @test_no_fold_nonzero(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>, %[[T1:.*]]: tensor<4xf32>, %[[T2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:         %[[VAL:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:         %[[RES:.*]] = "arithmeticops.addcmul"(%[[IN]], %[[VAL]], %[[T1]], %[[T2]]) : (tensor<4xf32>, f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:         return %[[RES]] : tensor<4xf32>
func.func @test_no_fold_nonzero(%in: tensor<4xf32>, %t1: tensor<4xf32>, %t2: tensor<4xf32>) -> tensor<4xf32> {
  %val = arith.constant 2.0 : f32
  %0 = "arithmeticops.addcmul"(%in, %val, %t1, %t2) : (tensor<4xf32>, f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// 8. dense<0> + value * (tensor1 * tensor2) -> value * (tensor1 * tensor2) (i32)
// CHECK-LABEL: func.func @test_rewrite_zero_input_int(
// CHECK-SAME:    %[[VAL:.*]]: i32, %[[T1:.*]]: tensor<4xi32>, %[[T2:.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         %[[SPLAT:.*]] = tensor.splat %[[VAL]] : tensor<4xi32>
// CHECK:         %[[PROD:.*]] = arithmeticops.mul %[[T1]] * %[[T2]] : tensor<4xi32>
// CHECK:         %[[RES:.*]] = arithmeticops.mul %[[SPLAT]] * %[[PROD]] : tensor<4xi32>
// CHECK:         return %[[RES]] : tensor<4xi32>
func.func @test_rewrite_zero_input_int(%val: i32, %t1: tensor<4xi32>, %t2: tensor<4xi32>) -> tensor<4xi32> {
  %zero_in = arith.constant dense<0> : tensor<4xi32>
  %0 = "arithmeticops.addcmul"(%zero_in, %val, %t1, %t2) : (tensor<4xi32>, i32, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// 9. dense<0.0> + value * (tensor1 * tensor2) -> value * (tensor1 * tensor2) (f32)
// CHECK-LABEL: func.func @test_rewrite_zero_input_float(
// CHECK-SAME:    %[[VAL:.*]]: f32, %[[T1:.*]]: tensor<4xf32>, %[[T2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         %[[SPLAT:.*]] = tensor.splat %[[VAL]] : tensor<4xf32>
// CHECK:         %[[PROD:.*]] = arithmeticops.mul %[[T1]] * %[[T2]] : tensor<4xf32>
// CHECK:         %[[RES:.*]] = arithmeticops.mul %[[SPLAT]] * %[[PROD]] : tensor<4xf32>
// CHECK:         return %[[RES]] : tensor<4xf32>
func.func @test_rewrite_zero_input_float(%val: f32, %t1: tensor<4xf32>, %t2: tensor<4xf32>) -> tensor<4xf32> {
  %zero_in = arith.constant dense<0.0> : tensor<4xf32>
  %0 = "arithmeticops.addcmul"(%zero_in, %val, %t1, %t2) : (tensor<4xf32>, f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// 10. input + 1 * (tensor1 * tensor2) -> input + (tensor1 * tensor2) (i32)
// CHECK-LABEL: func.func @test_rewrite_one_value_int(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xi32>, %[[T1:.*]]: tensor<4xi32>, %[[T2:.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         %[[PROD:.*]] = arithmeticops.mul %[[T1]] * %[[T2]] : tensor<4xi32>
// CHECK:         %[[RES:.*]] = "arithmeticops.add"(%[[IN]], %[[PROD]]) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK:         return %[[RES]] : tensor<4xi32>
func.func @test_rewrite_one_value_int(%in: tensor<4xi32>, %t1: tensor<4xi32>, %t2: tensor<4xi32>) -> tensor<4xi32> {
  %one = arith.constant 1 : i32
  %0 = "arithmeticops.addcmul"(%in, %one, %t1, %t2) : (tensor<4xi32>, i32, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// 11. input + 1.0 * (tensor1 * tensor2) -> input + (tensor1 * tensor2) (f32)
// CHECK-LABEL: func.func @test_rewrite_one_value_float(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>, %[[T1:.*]]: tensor<4xf32>, %[[T2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         %[[PROD:.*]] = arithmeticops.mul %[[T1]] * %[[T2]] : tensor<4xf32>
// CHECK:         %[[RES:.*]] = "arithmeticops.add"(%[[IN]], %[[PROD]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:         return %[[RES]] : tensor<4xf32>
func.func @test_rewrite_one_value_float(%in: tensor<4xf32>, %t1: tensor<4xf32>, %t2: tensor<4xf32>) -> tensor<4xf32> {
  %one = arith.constant 1.0 : f32
  %0 = "arithmeticops.addcmul"(%in, %one, %t1, %t2) : (tensor<4xf32>, f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// 12. input + value * (dense<1> * tensor2) -> input + (value * tensor2) (i32)
// CHECK-LABEL: func.func @test_rewrite_one_tensor1_int(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xi32>, %[[VAL:.*]]: i32, %[[T2:.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         %[[SPLAT:.*]] = tensor.splat %[[VAL]] : tensor<4xi32>
// CHECK:         %[[PROD:.*]] = arithmeticops.mul %[[SPLAT]] * %[[T2]] : tensor<4xi32>
// CHECK:         %[[RES:.*]] = "arithmeticops.add"(%[[IN]], %[[PROD]]) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK:         return %[[RES]] : tensor<4xi32>
func.func @test_rewrite_one_tensor1_int(%in: tensor<4xi32>, %val: i32, %t2: tensor<4xi32>) -> tensor<4xi32> {
  %one_t = arith.constant dense<1> : tensor<4xi32>
  %0 = "arithmeticops.addcmul"(%in, %val, %one_t, %t2) : (tensor<4xi32>, i32, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// 13. input + value * (dense<1.0> * tensor2) -> input + (value * tensor2) (f32)
// CHECK-LABEL: func.func @test_rewrite_one_tensor1_float(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>, %[[VAL:.*]]: f32, %[[T2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         %[[SPLAT:.*]] = tensor.splat %[[VAL]] : tensor<4xf32>
// CHECK:         %[[PROD:.*]] = arithmeticops.mul %[[SPLAT]] * %[[T2]] : tensor<4xf32>
// CHECK:         %[[RES:.*]] = "arithmeticops.add"(%[[IN]], %[[PROD]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:         return %[[RES]] : tensor<4xf32>
func.func @test_rewrite_one_tensor1_float(%in: tensor<4xf32>, %val: f32, %t2: tensor<4xf32>) -> tensor<4xf32> {
  %one_t = arith.constant dense<1.0> : tensor<4xf32>
  %0 = "arithmeticops.addcmul"(%in, %val, %one_t, %t2) : (tensor<4xf32>, f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// 14. input + value * (tensor1 * dense<1>) -> input + (value * tensor1) (i32)
// CHECK-LABEL: func.func @test_rewrite_one_tensor2_int(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xi32>, %[[VAL:.*]]: i32, %[[T1:.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         %[[SPLAT:.*]] = tensor.splat %[[VAL]] : tensor<4xi32>
// CHECK:         %[[PROD:.*]] = arithmeticops.mul %[[SPLAT]] * %[[T1]] : tensor<4xi32>
// CHECK:         %[[RES:.*]] = "arithmeticops.add"(%[[IN]], %[[PROD]]) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK:         return %[[RES]] : tensor<4xi32>
func.func @test_rewrite_one_tensor2_int(%in: tensor<4xi32>, %val: i32, %t1: tensor<4xi32>) -> tensor<4xi32> {
  %one_t = arith.constant dense<1> : tensor<4xi32>
  %0 = "arithmeticops.addcmul"(%in, %val, %t1, %one_t) : (tensor<4xi32>, i32, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// 15. input + value * (tensor1 * dense<1.0>) -> input + (value * tensor1) (f32)
// CHECK-LABEL: func.func @test_rewrite_one_tensor2_float(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>, %[[VAL:.*]]: f32, %[[T1:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT:     arithmeticops.addcmul
// CHECK:         %[[SPLAT:.*]] = tensor.splat %[[VAL]] : tensor<4xf32>
// CHECK:         %[[PROD:.*]] = arithmeticops.mul %[[SPLAT]] * %[[T1]] : tensor<4xf32>
// CHECK:         %[[RES:.*]] = "arithmeticops.add"(%[[IN]], %[[PROD]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:         return %[[RES]] : tensor<4xf32>
func.func @test_rewrite_one_tensor2_float(%in: tensor<4xf32>, %val: f32, %t1: tensor<4xf32>) -> tensor<4xf32> {
  %one_t = arith.constant dense<1.0> : tensor<4xf32>
  %0 = "arithmeticops.addcmul"(%in, %val, %t1, %one_t) : (tensor<4xf32>, f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}