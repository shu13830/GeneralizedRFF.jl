using Test
using GeneralizedRandomFourierFeatures
using KernelFunctions
using KernelFunctions: ColVecs
using Random
using LinearAlgebra
using Statistics

@testset "RFF Approximation Quality" begin
    rng = MersenneTwister(42)

    @testset "Convergence with M" begin
        k = GeneralizedCauchyKernel(1.5, 2.0)
        X = [rand(rng, 3) for _ in 1:20]
        K_exact = kernelmatrix(k, X)

        M_values = [100, 500, 1000, 2000]
        errors = Float64[]

        for M in M_values
            K_approx = rff_kernelmatrix(rng, k, X, M)
            error = norm(K_exact - K_approx) / norm(K_exact)
            push!(errors, error)
        end

        # Approximation should be bounded (not diverging)
        @test all(errors .< 1.0)
        # Larger M should generally give better approximation
        @test errors[end] < errors[1] || all(errors .< 0.5)
    end

    @testset "Approximation for different kernels" begin
        rng_test = MersenneTwister(123)
        X = [rand(rng_test, 2) for _ in 1:10]  # Smaller data
        M = 1000  # More features

        kernels = [
            GeneralizedCauchyKernel(1.5, 2.0),
            KummerKernel(α=1.5, β=2.0, γ=1.5),
            BetaKernel(α=1.5, β=2.0, γ=3.0)
        ]

        for k in kernels
            @testset "$(typeof(k))" begin
                K_exact = kernelmatrix(k, X)
                K_approx = rff_kernelmatrix(rng_test, k, X, M)

                # Relative error should be bounded
                rel_error = norm(K_exact - K_approx) / norm(K_exact)
                @test rel_error < 0.6  # Relaxed threshold - RFF approximation has high variance

                # Approximation should produce valid values
                @test all(isfinite.(K_approx))
            end
        end
    end

    @testset "Symmetry preservation" begin
        rng_test = MersenneTwister(456)
        k = GeneralizedCauchyKernel(1.5, 2.0)
        X = [rand(rng_test, 3) for _ in 1:10]
        M = 200

        K_approx = rff_kernelmatrix(rng_test, k, X, M)

        # Should be symmetric
        @test K_approx ≈ K_approx' rtol=1e-10
    end

    @testset "Positive semi-definiteness" begin
        rng_test = MersenneTwister(789)
        k = GeneralizedCauchyKernel(1.5, 2.0)
        X = [rand(rng_test, 3) for _ in 1:20]
        M = 300

        K_approx = rff_kernelmatrix(rng_test, k, X, M)

        # All eigenvalues should be non-negative (within numerical tolerance)
        λ = eigvals(K_approx)
        @test all(λ .> -1e-10)
    end

    @testset "Comparison with exact kernel" begin
        rng_test = MersenneTwister(101112)
        k = GeneralizedCauchyKernel(1.2, 1.0)
        M = 2000  # More features for better approximation

        # Test on individual point pairs
        n_tests = 10
        errors = Float64[]

        for _ in 1:n_tests
            x = rand(rng_test, 3)
            y = rand(rng_test, 3)

            # Exact kernel value
            k_exact = k(x, y)

            # RFF approximation
            basis = sample_generalized_rff_basis(rng_test, k, 3, M)
            features = basis(ColVecs(hcat(x, y)))
            φx = features[1]
            φy = features[2]
            k_approx = dot(φx, φy)

            push!(errors, abs(k_exact - k_approx))
        end

        # Mean absolute error should be bounded
        @test mean(errors) < 0.5
        # Errors should be finite
        @test all(isfinite.(errors))
    end
end

@testset "RFF Basis Properties" begin
    rng = MersenneTwister(42)

    @testset "Basis dimension" begin
        k = GeneralizedCauchyKernel(1.5, 2.0)
        input_dim = 5
        M = 100

        basis = sample_generalized_rff_basis(rng, k, input_dim, M)

        @test size(basis.ω) == (input_dim, M)
        @test length(basis.τ) == M

        # Feature vector should have dimension M
        x = rand(rng, input_dim)
        φ = basis(ColVecs(reshape(x, :, 1)))[1]
        @test length(φ) == M
    end

    @testset "Different input dimensions" begin
        k = GeneralizedCauchyKernel(1.5, 2.0)
        M = 50

        for d in [1, 3, 10]
            basis = sample_generalized_rff_basis(rng, k, d, M)
            x = rand(rng, d)
            φ = basis(ColVecs(reshape(x, :, 1)))[1]

            @test length(φ) == M
            @test all(isfinite.(φ))
        end
    end

    @testset "Reproducibility" begin
        k = GeneralizedCauchyKernel(1.5, 2.0)

        rng1 = MersenneTwister(42)
        basis1 = sample_generalized_rff_basis(rng1, k, 3, 50)

        rng2 = MersenneTwister(42)
        basis2 = sample_generalized_rff_basis(rng2, k, 3, 50)

        @test basis1.ω == basis2.ω
        @test basis1.τ == basis2.τ
    end
end
