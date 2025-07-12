# -----------------------------------------------------------------------------
# Generalized Cauchy Kernel
# -----------------------------------------------------------------------------
"""
    GeneralizedCauchyKernel(α, β)

Generalized Cauchy kernel.

# Defenition

For inputs ``x, x'``, exponent parameter ``\\alpha\\in(0,2]`` and ``\\beta>0``, the generalized Cauchy kernel is defined as
```math
k(x, x') = \\frac{1}{\\bigl(1 + \\|x - x'\\|^{\\alpha} / 2\\beta \\bigr)^{\\beta}
```
parameters for generalized RFF approximation: (R ~ Gamma(β, 1), λ = 1/(2β), α)
When ``β = 1``, it reduces to the Cauchy kernel.
When ``β → ∞``, it approaches the Exponential Power Kernel.
"""
struct GeneralizedCauchyKernel{M} <: KernelFunctions.Kernel
    α::Float64    # exponent parameter
    β::Float64    # tail parameter
    metric::M
    function GeneralizedCauchyKernel(α::T1, β::T2) where {T1<:Real, T2<:Real}
        0 < α <= 2 || throw(ArgumentError("α must be in (0,2] for PD"))
        0 < β || throw(ArgumentError("β must be positive"))
        new{T}(Float64(α), Float64(β), KernelFunctions.Euclidean())
    end
end

KernelFunctions.kappa(k::GeneralizedCauchyKernel, d::Real) = 1 / (1 + d^only(k.α) / (2*only(k.β)) )^only(k.β)

KernelFunctions.metric(k::GeneralizedCauchyKernel) = k.metric

function Base.show(io::IO, k::GeneralizedCauchyKernel)
    return print(io, "Generalized Cauchy Kernel (α = ", only(k.α),", β = ", only(k.β),", metric = ", k.metric, ")")
end

"""

Spectral mixing distribution R for a generalized Cauchy kernel
R ∼ Gamma(shape=β, scale=1)    
"""
function spectral_mixing_distribution(k::GeneralizedCauchyKernel)
    return Gamma(only(k.β), 1.)
end

"""Spectral parameters (α, λ=1/2β) for a generalized Cauchy kernel"""
function spectral_params(k::GeneralizedCauchyKernel)
    α = only(k.α)
    λ = 1.0 / (2*only(k.β))
    return (α, λ)
end

isa_kernel_for_genlrff(k::GeneralizedCauchyKernel) = true

function spectral_weights(::GeneralizedCauchyKernel)
    return 1.0, 1.0
end
