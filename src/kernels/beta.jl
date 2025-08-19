# -----------------------------------------------------------------------------
# Beta Kernel
# -----------------------------------------------------------------------------
"""
    BetaKernel(α, β)

Beta Kernel via spectral Beta mixture:
```math
k(r) = B(α, β)\\,U\\bigl(α,\\,1-β,\\,(r/ℓ)^2\\bigr)
```
where U is Tricomi's confluent hypergeometric function.
"""
struct BetaKernel{M} <: KernelFunctions.Kernel
    α::Float64    # exponent parameter (0 < α <= 2)
    β::Float64    # 1st parameter of Beta distribution
    γ::Float64    # 2nd parameter of Beta distribution
    metric::M
    function BetaKernel(;α::T1, β::T2, γ::T3) where {T1<:Real, T2<:Real, T3<:Real}
        0 < α <= 2 || throw(ArgumentError("α must be in (0,2] for PD"))
        0 < β || throw(ArgumentError("β must be positive"))
        0 < γ || throw(ArgumentError("γ must be positive"))
        metric = KernelFunctions.Euclidean()
        new{typeof(metric)}(Float64(α), Float64(β), Float64(γ), metric)
    end
end

function KernelFunctions.kappa(k::BetaKernel, d::Real)
    α = only(k.α)
    β = only(k.β)
    γ = only(k.γ)
    nominator = beta(β + d^α, γ)
    denominator = beta(β, γ)
    return nominator / denominator
end

KernelFunctions.metric(k::BetaKernel) = k.metric

function Base.show(io::IO, k::BetaKernel)
    return print(io, "Beta Kernel (α = ", only(k.α),", β = ", only(k.β)," γ =", only(k.γ)," metric = ", k.metric, ")")
end



"""
Spectral mixing distribution R for a Beta kernel: 
R ∼ Beta-exponential(β, γ)
where -log(R) follows a Beta distribution with parameters β and γ.
"""
function spectral_mixing_distribution(k::BetaKernel)
    β = only(k.β)
    γ = only(k.γ)
    neglog_bij = Bijectors.Chain(Bijectors.log, Bijectors.Scale(-1.0))
    return transformed(Beta(β, γ), neglog_bij)
end

"""Spectral parameters (α, λ=1) for a Beta kernel"""
function spectral_params(k::BetaKernel)
    α = only(k.α)
    λ = 1.
    return (α, λ)
end

isa_kernel_for_genlrff(::BetaKernel) = k.metric isa KernelFunctions.Euclidean

function spectral_weights(::BetaKernel)
    return 1.0, 1.0
end
