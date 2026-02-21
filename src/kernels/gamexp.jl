# -----------------------------------------------------------------------------
# Exponential Power Kernel
# -----------------------------------------------------------------------------
"""
    GammaExponentialKernel(; γ, metric=KernelFunctions.Euclidean())

Gamma Exponential (Exponential Power) Kernel.

# Definition

For inputs ``x, x'`` and exponent parameter ``\\gamma \\in (0,2]``, the kernel is defined as:
```math
k(x, x') = \\exp(-\\|x - x'\\|^{\\gamma})
```

This is also known as Exponential Power, Generalized Gaussian, or Subbotin Kernel.

# Parameters for Generalized RFF
- R = 1 (constant)
- λ = 1
- α = γ

# Special Cases
- `γ = 2`: Reduces to the Gaussian (RBF) kernel
- `γ = 1`: Reduces to the Laplacian kernel

# Note
This kernel is positive definite for `0 < γ ≤ 2`.
"""
struct GammaExponentialKernel{T<:Real,M} <: KernelFunctions.SimpleKernel
    γ::Vector{T}    # exponent parameter (0 < γ ≤ 2)
    metric::M
    function GammaExponentialKernel(; γ::T, metric=KernelFunctions.Euclidean()) where {T<:Real}
        0 < γ <= 2 || throw(ArgumentError("γ must be in (0,2] for positive definiteness"))
        new{T,typeof(metric)}([γ], metric)
    end
end

# Kernel evaluation
function KernelFunctions.kappa(k::GammaExponentialKernel, d::Real)
    γ = only(k.γ)
    return exp(-d^γ)
end

KernelFunctions.metric(k::GammaExponentialKernel) = k.metric

function Base.show(io::IO, k::GammaExponentialKernel)
    return print(io, "Gamma Exponential Kernel (γ = ", only(k.γ), ", metric = ", k.metric, ")")
end

# Aliases
const ExponentialPowerKernel = GammaExponentialKernel
const GeneralizedGaussianKernel = GammaExponentialKernel
const SubbotinKernel = GammaExponentialKernel

function isa_kernel_for_genlrff(κ::GammaExponentialKernel)
    γ = only(κ.γ)
    if 0 < γ <= 2.0
        if κ.metric isa KernelFunctions.Euclidean
            return true
        else
            return false
        end
    else
        return false
    end
end


"""
Spectral mixing distribution R for a gamma exponential kernel:
R = Constant 1
"""
function spectral_mixing_distribution(k::GammaExponentialKernel)
    return Dirac(1.0)  # Constant distribution returning 1
end

"""Spectral parameters (α, λ=1) for a gamma exponential kernel"""
function spectral_params(k::GammaExponentialKernel)
    α = only(k.γ)
    λ = 1.
    return (α, λ)
end

function spectral_weights(::GammaExponentialKernel)
    return 1.0, 1.0
end

