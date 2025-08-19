# -----------------------------------------------------------------------------
# Tricomi Kernel
# -----------------------------------------------------------------------------
"""
Tricomi Kernel via Beta-exponential mixture:
    k(r) = U(α, α - β + 1, (r/ℓ)^2)
where U is Tricomi's confluent hypergeometric function
"""
struct TricomiKernel{M} <: KernelFunctions.Kernel
    α::Float64
    β::Float64
    γ::Float64
    metric::M
    function TricomiKernel(;α::T1, β::T2, γ::T3) where {T1<:Real, T2<:Real, T3<:Real}
        0 < α <= 2 || throw(ArgumentError("α must be in (0,2] for PD"))
        0 < β || throw(ArgumentError("β must be positive"))
        0 < γ || throw(ArgumentError("γ must be positive"))
        metric = KernelFunctions.Euclidean()
        new{typeof(metric)}(Float64(α), Float64(β), Float64(γ), metric)
    end
end

function KernelFunctions.kappa(k::TricomiKernel, d::Real)
    α = only(k.α)
    β = only(k.β)
    γ = only(k.γ)
    return HypergeometricFunctions.U(β, 1 - γ, γ/β * d^α) * gamma(β+γ) / gamma(γ)
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

isa_kernel_for_genlrff(::TricomiKernel) = k.metric isa KernelFunctions.Euclidean

function spectral_weights(::TricomiKernel)
    return 1.0, 1.0
end
