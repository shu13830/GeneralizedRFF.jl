module GeneralizedRFF

# GeneralizedRFF.jl
# A Julia package extending RandomFourierFeatures.jl with support for a broad class of isotropic kernels via spectral mixture methods.
# see "A spectral mixture representation of isotropic kernels to generalize random Fourier features" (https://arxiv.org/abs/2411.02770) for details.

using AbstractGPs
using BayesianLinearRegressors: BayesianLinearRegressor
using Bijectors
using RandomFourierFeatures
using RandomFourierFeatures: BasisFunctionRegressor
using Distributions
using KernelFunctions
using Random, LinearAlgebra
using SpecialFunctions: besselk, gamma, beta, loggamma, logbeta
import HypergeometricFunctions

include("genl_random_fourier_features.jl")
include("symalphastable.jl")
include("kernels/gamexp.jl")
include("kernels/genlcauchy.jl")
include("kernels/matern.jl")
include("kernels/beta.jl")
include("kernels/kummer.jl")
include("kernels/tricomi.jl")
include("compat.jl")

# Export public API
export sample_generalized_rff_basis, rff_kernelmatrix
# Note: GammaExponentialKernel not exported to avoid conflict with KernelFunctions.jl
# Use GeneralizedRFF.GammaExponentialKernel if needed
export SubbotinKernel, GeneralizedGaussianKernel, ExponentialPowerKernel,
    GeneralizedCauchyKernel, BetaKernel, KummerKernel, TricomiKernel

end # module