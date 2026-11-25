// RUN: arithmetic-opt %s --fuse-arithmetic-ops | FileCheck %s

// 1. Positive test: single-use add + mul -> fused_add_mul (scalar f32)
// CHECK-LABEL: func.func @test_fuse_scalar
func.func @test_fuse_scalar(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-NOT: "arithmeticops.add"
  // CHECK-NOT: arithmeticops.mul
  // CHECK: %[[RES:.*]] = "arithmeticops.fused_add_mul"(%arg0, %arg1, %arg2) : (f32, f32, f32) -> f32
  // CHECK: return %[[RES]] : f32
  %0 = "arithmeticops.add"(%a, %b) : (f32, f32) -> f32
  %1 = arithmeticops.mul %0 * %c : f32
  return %1 : f32
}

// 2. Commutative test: c * (a + b) -> fused_add_mul(a, b, c)
// CHECK-LABEL: func.func @test_fuse_commutative
func.func @test_fuse_commutative(%a: i32, %b: i32, %c: i32) -> i32 {
  // CHECK-NOT: "arithmeticops.add"
  // CHECK-NOT: arithmeticops.mul
  // CHECK: %[[RES:.*]] = "arithmeticops.fused_add_mul"(%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
  // CHECK: return %[[RES]] : i32
  %0 = "arithmeticops.add"(%a, %b) : (i32, i32) -> i32
  %1 = arithmeticops.mul %c * %0 : i32
  return %1 : i32
}

// 3. Negative test: multi-use add should NOT fuse
// CHECK-LABEL: func.func @test_no_fuse_multi_use
func.func @test_no_fuse_multi_use(%a: f32, %b: f32, %c: f32) -> (f32, f32) {
  // CHECK: %[[SUM:.*]] = "arithmeticops.add"(%arg0, %arg1) : (f32, f32) -> f32
  // CHECK: %[[PROD:.*]] = arithmeticops.mul %[[SUM]] * %arg2 : f32
  // CHECK-NOT: "arithmeticops.fused_add_mul"
  // CHECK: return %[[SUM]], %[[PROD]] : f32, f32
  %0 = "arithmeticops.add"(%a, %b) : (f32, f32) -> f32
  %1 = arithmeticops.mul %0 * %c : f32
  return %0, %1 : f32, f32
}

// 4. Executable main to verify numerical output: (2.0 + 3.0) * 4.0 = 20.0
func.func @main() {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %c = arith.constant 4.0 : f32
  %res = func.call @test_fuse_scalar(%a, %b, %c) : (f32, f32, f32) -> f32
  vector.print %res : f32
  return
}