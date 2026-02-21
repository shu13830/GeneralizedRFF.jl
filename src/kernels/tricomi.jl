# -----------------------------------------------------------------------------
# Tricomi Kernel
# -----------------------------------------------------------------------------
"""
Tricomi Kernel via Beta-exponential mixture:
    k(r) = U(α, α - β + 1, (r/ℓ)^2)
where U is Tricomi's confluent hypergeometric function
"""
struct TricomiKernel{T<:Real,M} <: KernelFunctions.Kernel
    α::Vector{T}
    β::Vector{T}
    γ::Vector{T}
    metric::M
    function TricomiKernel(;α::T1, β::T2, γ::T3) where {T1<:Real, T2<:Real, T3<:Real}
        0 < α <= 2 || throw(ArgumentError("α must be in (0,2] for PD"))
        0 < β || throw(ArgumentError("β must be positive"))
        0 < γ || throw(ArgumentError("γ must be positive"))
        T = promote_type(T1, T2, T3)
        metric = KernelFunctions.Euclidean()
        new{T,typeof(metric)}([T(α)], [T(β)], [T(γ)], metric)
    end
end

function KernelFunctions.kappa(k::TricomiKernel, d::Real)
    α = only(k.α)
    β = only(k.β)
    γ = only(k.γ)
    # Use loggamma for numerical stability with large parameters
    log_gamma_ratio = loggamma(β + γ) - loggamma(γ)
    gamma_ratio = exp(log_gamma_ratio)
    return HypergeometricFunctions.U(β, 1 - γ, γ/β * d^α) * gamma_ratio
end

KernelFunctions.metric(k::TricomiKernel) = k.metric

function Base.show(io::IO, k::TricomiKernel)
    return print(io, "Tricomi Kernel (α = ", only(k.α), ", β = ", only(k.β), ", γ = ", only(k.γ), ", metric = ", k.metric, ")")
end

"""
Spectral mixing distribution R for a Tricomi kernel
    R ∼ F(2β, 2γ)
"""
function spectral_mixing_distribution(k::TricomiKernel)
    β = only(k.β)
    γ = only(k.γ)
    return FDist(2*β, 2*γ)
end

"""Spectral parameters (α, λ=1) for a Tricomi kernel"""
function spectral_params(k::TricomiKernel)
    α = only(k.α)
    return (α, 1.0)
end

isa_kernel_for_genlrff(k::TricomiKernel) = k.metric isa KernelFunctions.Euclidean

function spectral_weights(::TricomiKernel)
    return 1.0, 1.0
end
