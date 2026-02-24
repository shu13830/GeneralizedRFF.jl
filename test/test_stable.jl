using Test
using GeneralizedRandomFourierFeatures
using Random
using Statistics

@testset "Symmetric Alpha-Stable Sampling" begin
    rng = MersenneTwister(42)

    @testset "α = 2 (Gaussian case)" begin
        # For α = 2, Sα should follow √2 * N(0, I)
        α = 2.0
        d = 100
        n_samples = 10000

        samples = zeros(n_samples, d)
        for i in 1:n_samples
            samples[i, :] = GeneralizedRandomFourierFeatures.sample_Sα(rng, α, d)
        end

        # Expected: mean ≈ 0, variance ≈ 2
        sample_mean = mean(samples)
        sample_var = var(samples)

        @test abs(sample_mean) < 0.1
        @test abs(sample_var - 2.0) < 0.2
    end

    @testset "α = 1 (Cauchy case)" begin
        # For α = 1, Sα follows Cauchy distribution
        # Cauchy has undefined mean and variance, but median = 0
        α = 1.0
        d = 1
        n_samples = 1000

        samples = [GeneralizedRandomFourierFeatures.sample_Sα(rng, α, d)[1] for _ in 1:n_samples]

        # Check median is close to 0
        @test abs(median(samples)) < 0.5

        # Check heavy tails: should have large absolute values
        @test maximum(abs.(samples)) > 5.0
    end

    @testset "Output dimension" begin
        α = 1.5
        for d in [1, 5, 10]
            s = GeneralizedRandomFourierFeatures.sample_Sα(rng, α, d)
            @test length(s) == d
        end
    end

    @testset "Numerical stability for α ≈ 2" begin
        # Should not produce NaN or Inf
        # Note: α must be in (0, 2] for positive definiteness
        α_values = [1.9, 1.99, 1.999, 2.0]
        d = 5

        for α in α_values
            s = GeneralizedRandomFourierFeatures.sample_Sα(rng, α, d)
            @test all(isfinite.(s))
        end
    end

    @testset "Different α values" begin
        # Test that different α produce different distributions
        d = 100
        n_samples = 1000

        function sample_moments(α)
            samples = [GeneralizedRandomFourierFeatures.sample_Sα(rng, α, d) for _ in 1:n_samples]
            # Use robust statistics
            mads = [median(abs.(s .- median(s))) for s in samples]
            return median(mads)
        end

        # α closer to 0 should produce heavier tails (larger MAD)
        mad_05 = sample_moments(0.5)
        mad_15 = sample_moments(1.5)
        mad_20 = sample_moments(2.0)

        # This is a rough check - exact relationship is complex
        @test mad_05 > mad_20
    end
end

@testset "Spectral Sampling Integration" begin
    rng = MersenneTwister(123)

    @testset "Generalized Cauchy" begin
        k = GeneralizedCauchyKernel(1.5, 2.0)
        basis = sample_generalized_rff_basis(rng, k, 3, 100)

        @test size(basis.ω) == (3, 100)
        @test length(basis.τ) == 100
        @test all(isfinite.(basis.ω))
        @test all(isfinite.(basis.τ))
    end

    @testset "Kummer" begin
        k = KummerKernel(α=1.5, β=2.0, γ=1.5)
        basis = sample_generalized_rff_basis(rng, k, 2, 50)

        @test size(basis.ω) == (2, 50)
        @test all(isfinite.(basis.ω))
    end
end
