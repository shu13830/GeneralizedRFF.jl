# -----------------------------------------------------------------------------
# Kummer Kernel
# -----------------------------------------------------------------------------
"""
Kummer Kernel via spectral F-distribution mixture:
  k(r) = U(α, β, (r/ℓ)^2)
where U is confluent hypergeometric U function
"""
struct KummerKernel{T<:Real,M} <: KernelFunctions.Kernel
    α::Vector{T}
    β::Vector{T}
    γ::Vector{T}
    metric::M
    function KummerKernel(;α::T1, β::T2, γ::T3) where {T1<:Real, T2<:Real, T3<:Real}
        0 < α <= 2 || throw(ArgumentError("α must be in (0,2] for PD"))
        0 < β || throw(ArgumentError("β must be positive"))
        0 < γ || throw(ArgumentError("γ must be positive"))
        T = promote_type(T1, T2, T3)
        metric = KernelFunctions.Euclidean()
        new{T,typeof(metric)}([T(α)], [T(β)], [T(γ)], metric)
    end
end

function KernelFunctions.kappa(k::KummerKernel, d::Real)
    α = only(k.α)
    β = only(k.β)
    γ = only(k.γ)
    return HypergeometricFunctions.M(β, β+γ, -d^α)
end

KernelFunctions.metric(k::KummerKernel) = k.metric

function Base.show(io::IO, k::KummerKernel)
    return print(io, "Kummer Kernel (α = ", only(k.α), ", β = ", only(k.β), ", γ = ", only(k.γ), ", metric = ", k.metric, ")")
end

"""
Spectral mixing distribution R for a Kummer kernel
    R ∼ Beta(β, γ)
"""
function spectral_mixing_distribution(k::KummerKernel)
    β = only(k.β)
    γ = only(k.γ)
    return Beta(β, γ)
end

"""Spectral parameters (α, λ=1) for a Kummer kernel"""
function spectral_params(k::KummerKernel)
    α = only(k.α)
    return (α, 1.)
end

isa_kernel_for_genlrff(k::KummerKernel) = k.metric isa KernelFunctions.Euclidean

function spectral_weights(::KummerKernel)
    return 1.0, 1.0
end
