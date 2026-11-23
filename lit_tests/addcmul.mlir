// RUN: arithmetic-opt %s --convert-arithmeticops-to-arith | FileCheck %s

// CHECK-LABEL: func.func @test_addcmul_int(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xi32>, %[[VAL:.*]]: i32,
// CHECK-SAME:    %[[T1:.*]]: tensor<4xi32>, %[[T2:.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK:         %[[SPLAT:.*]] = tensor.splat %[[VAL]] : tensor<4xi32>
// CHECK:         %[[PROD:.*]] = arith.muli %[[T1]], %[[T2]] : tensor<4xi32>
// CHECK:         %[[SCALED:.*]] = arith.muli %[[SPLAT]], %[[PROD]] : tensor<4xi32>
// CHECK:         %[[RES:.*]] = arith.addi %[[IN]], %[[SCALED]] : tensor<4xi32>
// CHECK:         return %[[RES]] : tensor<4xi32>

func.func @test_addcmul_int(%in: tensor<4xi32>, %val: i32,
                            %t1: tensor<4xi32>, %t2: tensor<4xi32>) -> tensor<4xi32> {
  %0 = "arithmeticops.addcmul"(%in, %val, %t1, %t2) : (tensor<4xi32>, i32, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func.func @test_addcmul_float(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>, %[[VAL:.*]]: f32,
// CHECK-SAME:    %[[T1:.*]]: tensor<4xf32>, %[[T2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:         %[[SPLAT:.*]] = tensor.splat %[[VAL]] : tensor<4xf32>
// CHECK:         %[[PROD:.*]] = arith.mulf %[[T1]], %[[T2]] : tensor<4xf32>
// CHECK:         %[[SCALED:.*]] = arith.mulf %[[SPLAT]], %[[PROD]] : tensor<4xf32>
// CHECK:         %[[RES:.*]] = arith.addf %[[IN]], %[[SCALED]] : tensor<4xf32>
// CHECK:         return %[[RES]] : tensor<4xf32>

func.func @test_addcmul_float(%in: tensor<4xf32>, %val: f32,
                              %t1: tensor<4xf32>, %t2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "arithmeticops.addcmul"(%in, %val, %t1, %t2) : (tensor<4xf32>, f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}