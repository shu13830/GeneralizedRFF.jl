# -----------------------------------------------------------------------------
# Exponential Power Kernel
# -----------------------------------------------------------------------------
"""
    GammaExponentialKernel(γ; metric=KernelFunctions.Euclidean)

    k(u) = exp(- u^γ )

This is also known as Exponential Power, Generalized Gaussian, or Subbotin Kernel
parameters for generalized RFF approximation: (R = 1, λ = 1, α = γ)
when `γ = 2`, this reduces to the Gaussian (RBF) kernel.
when `γ = 1`, it is the Laplacian kernel.
This kernel is positive definite for `0 < γ ≤ 2`.
"""
ExponentialPowerKernel = GammaExponentialKernel
GeneralizedGaussianKernel = GammaExponentialKernel
SubbotinKernel = GammaExponentialKernel

function isa_kernel_for_genlrff(κ::GammaExponentialKernel)
    if 0 < κ.γ <= 2.0
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
    return Categorical([1.0], [1.0])  # Constant distribution
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

