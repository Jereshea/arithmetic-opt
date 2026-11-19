module{
    func.func @test_weighted_add_with_bias(%in: tensor<2x3x4xi32>, %bias: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    // Case 1: With optional $bias
    %case_1 = arithmeticops.weighted_add %in, %bias testAttr = [1.5 : f32, 2.0 : f32, 0.5 : f32]
        : tensor<2x3x4xi32>, tensor<2x3x4xf32> -> tensor<2x3x4xf32>

    return %case_1 : tensor<2x3x4xf32>
    }

    func.func @test_weighted_add_without_bias(%in: tensor<2x3x4xi32>, %bias: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    // Case 2: Without optional $bias 
    %case_2 = arithmeticops.weighted_add %in testAttr = [1.0 : f32, 2.0 : f32, 3.0 : f32]
        : tensor<2x3x4xi32> -> tensor<2x3x4xf32>

    return %case_2 : tensor<2x3x4xf32>
    }
}