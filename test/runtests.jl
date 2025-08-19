using Test
using RandomFourierFeatures
using KernelFunctions
using GeneralizedRFFs

# Set random seed for reproducibility
global_rng = MersenneTwister(1234)

# Define small input data for testing
X = [rand(global_rng, 3) for _ in 1:5]
Y = [rand(global_rng, 3) for _ in 1:5]

# Define number of features for approximation
M = 2000

# Helper to approximate kernel by random features
function approximate_kernel(rng, k, x, y, M)
    φ = sample_generalized_rff_basis(rng, k, length(x), M)
    # Compute feature map values
    fx = φ(x)                # returns vector of size M*2
    fy = φ(y)
    return dot(fx, fy)       # approximate kernel via inner product
end

# Test RBF fallback behavior
@testset "RBF Fallback" begin
    k = SqExponentialKernel(1.5)
    # Using core library vs generalized
    φ1 = sample_rff_basis(global_rng, k, 3, M)
    φ2 = sample_generalized_rff_basis(global_rng, k, 3, M)
    @test typeof(φ1) == typeof(φ2)
    @test φ1.inner == φ2.inner
    @test φ1.outer_scaled == φ2.outer_scaled
end

# Test new kernels correctness
@testset "Extended Kernels" begin
    kernels = [ GeneralizedMaternKernel(1.0, 0.5),
                GeneralizedCauchyKernel(1.2, 1.0),
                ExponentialPowerKernel(0.8, 1.5) ]
    for k in kernels
        @testset string(typeof(k)) begin
            # Exact kernel matrix entry for two points
            exact = kernelmatrix(k, X[1]', X[2]')[]
            approx = approximate_kernel(global_rng, k, X[1], X[2], M)
            @test isapprox(approx, exact; atol=0.1)
        end
    end
end

# Test Beta, Kummer, Tricomi structure (ensure no errors)
@testset "Beta/Kummer/Tricomi Sampling" begin
    ks = [ BetaKernel(1.0, 2.0, 3.0),
           KummerKernel(1.0, 2.0, 0.5),
           TricomiKernel(1.0, 1.5, 0.7) ]
    for k in ks
        @test sample_generalized_rff_basis(global_rng, k, 2, 100) isa RandomFourierFeatures.RFFBasis
    end
end
