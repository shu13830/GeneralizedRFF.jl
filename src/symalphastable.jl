"""
    sample_Sα(rng::AbstractRNG, α::Float64, d::Int)

Sample a d-dimensional symmetric alpha-stable vector Sα using the Gaussian
scale mixture representation from Proposition 1.

# Mathematical Background
For α ∈ (0, 2], the symmetric α-stable random vector Sα can be represented as:
    Sα = √(2Aα) · N
where N ~ MvNormal(0, I) and Aα is a random scale factor.

# Special Cases
- α = 2: Sα = √2 · N (Gaussian case, A₂ = 1)
- α = 1: Sα follows a multivariate Cauchy distribution

# Arguments
- `rng`: Random number generator
- `α`: Stability parameter (0 < α ≤ 2)
- `d`: Dimension of the output vector

# Returns
- `Sα`: d-dimensional symmetric α-stable random vector

# Reference
Langrené, Warin & Gruet (2024), arXiv:2411.02770v3, Proposition 1
"""
function sample_Sα(rng::AbstractRNG, α::Float64, d::Int)
    # Special case: α ≈ 2 (Gaussian)
    # For numerical stability and efficiency, handle Gaussian case directly
    if abs(α - 2.0) < 1e-10
        return sqrt(2.0) .* randn(rng, d)
    end

    Aα = zeros(Float64, d)
    for i in 1:d
        U1 = rand(rng)  # Uniformly distributed in [0, 1)
        U2 = rand(rng)  # Uniformly distributed in [0, 1)
        # Generate a random angle θ uniformly in [-π/2, π/2)
        # and a random scale W from the exponential distribution
        W = -log(U1)
        θ = π * (U2 - 1/2)
        # Compute Aα using equation (15) from the paper
        numerator = sin(α*π/4 + α*θ/2)
        denominator = (cos(θ))^(2/α)
        cosine_term = (cos(α*π/4 + (α/2 - 1)*θ) / W)^((2/α) - 1)
        Aα[i] = numerator / denominator * cosine_term
    end
    # sample multivariate symmetric stable random vectors Sα
    # S_α = √(2Aα) * N, where N ~ MvNormal(0, I)
    Sα = sqrt.(2 .* Aα) .* randn(rng, d)  # Scale by sqrt(2Aα)
    return Sα
end
