"""
Sample a d-dimensional symmetric alpha-stable vector Sα
see Proposition 1 in "A spectral mixture representation of isotropic kernels to generalize random Fourier features" (https://arxiv.org/abs/2411.02770)
"""
function sample_Sα(rng::AbstractRNG, α::Float64, d::Int)
    Aα = zeros(Float64, d)
    for i in 1:d
        U1 = rand(rng)  # Uniformly distributed in [0, 1)
        U2 = rand(rng)  # Uniformly distributed in [0, 1)
        # Generate a random angle θ uniformly in [-π, π)
        # and a random scale W from the exponential distribution
        W = -log(U1)
        θ = π * (U2 - 1/2)
        # Compute Aα
        numerator = sin(α*π/4 + α*θ/2)
        denominator = (cos(θ))^(2/α)
        cosine_term = (cos(α*π/4 + (α/2 - 1)*θ) / W)^((2/α) - 1)
        Aα[i] = numerator / denominator * cosine_term
    end
    # sample multivariate symmetric stable random vectors Sα
    # S_α = √(2Aα) * N, where N ~ MvNormal(0, I)
    Sα = sqrt(2 * Aα) .* randn(rng, d)  # Scale by sqrt(2Aα)
    return Sα
end
