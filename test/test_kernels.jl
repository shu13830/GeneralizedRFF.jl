using Test
using GeneralizedRandomFourierFeatures
using KernelFunctions
using Random
using LinearAlgebra

@testset "Kernel Properties" begin
    @testset "GeneralizedCauchyKernel" begin
        k = GeneralizedCauchyKernel(1.5, 2.0)

        # Test positive definiteness at origin
        @test k([0.0, 0.0], [0.0, 0.0]) ≈ 1.0

        # Test symmetry
        x, y = [1.0, 2.0], [3.0, 4.0]
        @test k(x, y) ≈ k(y, x)

        # Test decay with distance
        @test k([0.0], [0.0]) > k([0.0], [1.0])
        @test k([0.0], [1.0]) > k([0.0], [2.0])

        # Test parameter constraints
        @test_throws ArgumentError GeneralizedCauchyKernel(2.5, 1.0)  # α > 2
        @test_throws ArgumentError GeneralizedCauchyKernel(1.5, -1.0)  # β < 0
    end

    @testset "KummerKernel" begin
        k = KummerKernel(α=1.5, β=2.0, γ=1.5)

        # Test positive definiteness at origin
        @test k([0.0], [0.0]) ≈ 1.0

        # Test symmetry
        x, y = [1.0], [2.0]
        @test k(x, y) ≈ k(y, x)

        # Test parameter constraints
        @test_throws ArgumentError KummerKernel(α=2.5, β=1.0, γ=1.0)
        @test_throws ArgumentError KummerKernel(α=1.5, β=-1.0, γ=1.0)
    end

    @testset "BetaKernel" begin
        k = BetaKernel(α=1.5, β=2.0, γ=3.0)

        # Test positive definiteness at origin
        @test k([0.0], [0.0]) ≈ 1.0

        # Test symmetry
        x, y = [1.0], [2.0]
        @test k(x, y) ≈ k(y, x)
    end

    @testset "TricomiKernel" begin
        k = TricomiKernel(α=1.5, β=2.0, γ=1.5)

        # Test symmetry
        x, y = [1.0], [2.0]
        @test k(x, y) ≈ k(y, x)

        # Test that kernel is finite
        @test isfinite(k([0.0], [1.0]))
    end
end

@testset "Kernel Matrix Properties" begin
    rng = MersenneTwister(1234)
    X = [rand(rng, 2) for _ in 1:10]

    kernels = [
        GeneralizedCauchyKernel(1.5, 2.0),
        KummerKernel(α=1.5, β=2.0, γ=1.5),
        BetaKernel(α=1.5, β=2.0, γ=3.0),
        TricomiKernel(α=1.5, β=2.0, γ=1.5)
    ]

    for k in kernels
        @testset "$(typeof(k))" begin
            K = kernelmatrix(k, X)

            # Test symmetry
            @test K ≈ K'

            # Test positive definiteness (all eigenvalues should be positive)
            λ = eigvals(K)
            @test all(λ .> -1e-10)  # Allow small numerical errors

            # Test diagonal elements
            @test all(diag(K) .> 0)
        end
    end
end

@testset "Special Cases" begin
    @testset "GammaExponentialKernel basic properties" begin
        # GammaExponentialKernel(γ=2) gives exp(-r^2), which differs from
        # SqExponentialKernel's exp(-r²/2) by a factor of 2 in the exponent.
        # Here we just verify basic kernel properties.
        k_ep = GeneralizedRandomFourierFeatures.GammaExponentialKernel(γ=2.0)

        x, y = [1.0], [2.0]
        # Kernel should be 1 at origin
        @test k_ep(x, x) ≈ 1.0
        # Kernel should be positive
        @test k_ep(x, y) > 0
        # Kernel should decay with distance
        @test k_ep(x, x) > k_ep(x, y)
        # For γ=2, k(x,y) = exp(-|x-y|²) = exp(-1) when |x-y|=1
        @test k_ep(x, y) ≈ exp(-1.0) rtol=1e-10
    end
end
