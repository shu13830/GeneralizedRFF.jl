module GeneralizedRFFs

# GeneralizedRFFs.jl
# A Julia package extending RandomFourierFeatures.jl with support for a broad class of isotropic kernels via spectral mixture methods.
# see "A spectral mixture representation of isotropic kernels to generalize random Fourier features" (https://arxiv.org/abs/2411.02770) for details.

using AbstractGPs
using Bijectors
using RandomFourierFeatures
using Distributions
using KernelFunctions
using Random, LinearAlgebra
using SpecialFunctions: besselk, gamma, beta
using HypergeometricFunctions: U, M

include("genl_random_fourier_features.jl")
include("symalphastable.jl")
include("kernels/gamexp.jl")
include("kernels/genlcauchy.jl")
include("kernels/matern.jl")
include("kernels/beta.jl")
include("kernels/kummer.jl") 
include("kernels/tricomi.jl")

# Export public API
export sample_generalized_rff_basis
export SubbotinKernel, GeneralizedGaussianKernel, ExponentialPowerKernel, 
    GeneralizedCauchyKernel, BetaKernel, KummerKernel, TricomiKernel

end # module