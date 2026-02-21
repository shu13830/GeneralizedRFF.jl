# The weight_space_approx function needed for ApproximateGPs.pathwise_sample
@doc raw"""
    build_grff_weight_space_approx(rng::AbstractRNG, input_dims::Integer, num_features::Integer)

Builds a closure `grff_weight_space_approx(f::AbstractGP)` which takes an
`AbstractGP` as input and constructs a Bayesian linear regression model which
approximates `f`. `f` is assumed to be a zero mean prior GP with one of the
kernels supported by this package.
"""
function build_grff_weight_space_approx(
    rng::Random.AbstractRNG, input_dims::Integer, num_features::Integer
)
    function grff_weight_space_approx(f::AbstractGPs.AbstractGP)
        f.mean isa AbstractGPs.ZeroMean ||
            error("The GP to be approximated must have zero mean")
        ϕ = sample_grff_basis(rng, f.kernel, input_dims, num_features)
        blr = BayesianLinearRegressor(Zeros(num_features), Diagonal(Ones(num_features)))
        return BasisFunctionRegressor(blr, ϕ)
    end
    return grff_weight_space_approx
end

function sample_grff_basis(kernel, input_dims::Integer, num_features=100::Integer)
    return sample_grff_basis(Random.GLOBAL_RNG, kernel, input_dims, num_features)
end

# Public API alias
const sample_generalized_rff_basis = sample_grff_basis


# Helper function to extract lengthscale from kernel
# Returns 1.0 for kernels without explicit lengthscale
function get_lengthscale(k::KernelFunctions.Kernel)
    if hasproperty(k, :ℓ)
        return getfield(k, :ℓ)
    else
        return 1.0  # Default lengthscale
    end
end

# -- Main API: sample generalized RFF basis --
"""
Sample an RFF basis for any supported isotropic kernel.

# Arguments
- `rng`: RNG instance (e.g., `MersenneTwister`)
- `k`: A `KernelFunctions.Kernel` subtype (supports RBF and extended kernels)
- `input_dims`: Dimensionality of input data
- `num_features`: Number of random features M

# Returns
A `RandomFourierFeatures.RFFBasis` object representing the random feature map.
"""
function sample_grff_basis(
    rng::AbstractRNG,
    k::KernelFunctions.Kernel,
    input_dims::Int,
    num_features::Int
)
    # If Gaussian (SqExponential), defer to built-in implementation
    if k isa KernelFunctions.SqExponentialKernel
        return RandomFourierFeatures.sample_rff_basis(rng, k, input_dims, num_features)
    end

    if !isa_kernel_for_genlrff(k)
        throw(ArgumentError("Unsupported kernel type for generalized RFF: $(typeof(k))"))
    end

    # Retrieve spectral parameters alpha and lambda
    α, λ = spectral_params(k)
    # Get mixing distribution for scale variable R
    dist_R = spectral_mixing_distribution(k)

    # Initialize containers for frequencies and phases
    ω = zeros(Float64, input_dims, num_features)
    τ = 2π * rand(rng, num_features)  # Uniform phases in [0, 2π)

    # Sample random frequencies for each feature
    for m in 1:num_features
        R_val = rand(rng, dist_R)             # Sample mixture scale
        S_vec = sample_Sα(rng, α, input_dims) # Sample stable direction
        # Compute frequency vector: ω = (λR)^(1/α) · Sα
        scale_factor = (λ * R_val)^(1/α)
        ω[:, m] = scale_factor * S_vec
    end

    # Extract the kernel's lengthscale ℓ (default 1.0)
    ℓ = get_lengthscale(k)

    # Return the constructed RFF basis
    return RandomFourierFeatures.RFFBasis(
        inner=ℓ,
        outer_scaled=λ,
        ω,
        τ)
end