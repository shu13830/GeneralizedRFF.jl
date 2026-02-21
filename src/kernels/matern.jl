# -----------------------------------------------------------------------------
# Matérn Kernel
# -----------------------------------------------------------------------------
"""
Spectral mixing distribution R for a generalized Mate\`rn kernel: 
R ∼ InverseGamma(ν, 1.)
"""
function spectral_mixing_distribution(k::MaternKernel)
    return InverseGamma(only(k.ν), 1.)
end

"""Spectral parameters (α, λ=ν/2) for a generalized Mate\`rn kernel"""
function spectral_params(k::MaternKernel)
    α = 2.0
    λ = only(k.ν) / 2
    return (α, λ)
end

isa_kernel_for_genlrff(k::MaternKernel) = k.metric isa KernelFunctions.Euclidean

function spectral_weights(::MaternKernel)
    return 1.0, 1.0
end
