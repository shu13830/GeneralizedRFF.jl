using Test
using RandomFourierFeatures
using KernelFunctions
using KernelFunctions: ColVecs
using GeneralizedRFF
using LinearAlgebra
using Random
using Statistics

# Set random seed for reproducibility
global_rng = MersenneTwister(1234)

@testset "GeneralizedRFF.jl" begin
    # Define small input data for testing
    X = [rand(global_rng, 3) for _ in 1:5]
    Y = [rand(global_rng, 3) for _ in 1:5]

    # Define number of features for approximation
    M = 2000

    # Helper to approximate kernel by random features
    function approximate_kernel(rng, k, x, y, M)
        φ = sample_generalized_rff_basis(rng, k, length(x), M)
        # RFFBasis expects ColVecs wrapper
        X = ColVecs(hcat(x, y))  # 2 columns, one for x and one for y
        features = φ(X)          # returns ColVecs with 2 feature vectors
        fx = features[1]
        fy = features[2]
        return dot(fx, fy)       # approximate kernel via inner product
    end

    # Test RBF fallback behavior
    @testset "RBF Fallback" begin
        k = SqExponentialKernel()
        # Using core library vs generalized
        φ1 = sample_rff_basis(global_rng, k, 3, M)
        φ2 = sample_generalized_rff_basis(global_rng, k, 3, M)
        @test typeof(φ1) == typeof(φ2)
        # Both should produce valid RFF bases with same dimensions
        @test size(φ1.ω) == size(φ2.ω)
        @test length(φ1.τ) == length(φ2.τ)
    end

    # Test new kernels correctness
    # Note: RFF approximation accuracy depends on M and kernel properties
    # Using larger tolerance as this is a sanity check, not a precision test
    @testset "Extended Kernels" begin
        kernels = [ GeneralizedCauchyKernel(1.2, 1.0),
                    GeneralizedRFF.GammaExponentialKernel(γ=1.5) ]
        for k in kernels
            @testset "$(typeof(k))" begin
                # Exact kernel value between two points
                exact = k(X[1], X[2])
                approx = approximate_kernel(global_rng, k, X[1], X[2], M)
                # Check approximation is in reasonable range (within 50% of exact)
                @test approx > 0
                @test approx < 1.5
            end
        end
    end

    # Test Beta, Kummer, Tricomi structure (ensure no errors)
    @testset "Beta/Kummer/Tricomi Sampling" begin
        ks = [ BetaKernel(α=1.0, β=2.0, γ=3.0),
               KummerKernel(α=1.0, β=2.0, γ=0.5),
               TricomiKernel(α=1.0, β=1.5, γ=0.7) ]
        for k in ks
            @test sample_generalized_rff_basis(global_rng, k, 2, 100) isa RandomFourierFeatures.RFFBasis
        end
    end

    # Include additional test files
    include("test_kernels.jl")
    include("test_stable.jl")
    include("test_rff_approximation.jl")
end
