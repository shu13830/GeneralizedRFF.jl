# -----------------------------------------------------------------------------
# Self-contained RFFBasis implementation
# Replaces dependency on RandomFourierFeatures.jl
# -----------------------------------------------------------------------------

"""
    RFFBasis{Tinner,Touter,Tω,Tτ,Tsample}

Random Fourier Feature basis for kernel approximation.

Stores sampled frequencies `ω` and phases `τ`, and computes the feature map
`ϕ(x) = outer_weights * cos(ω' * (x / inner_weights) + τ)`.

# Fields
- `inner_weights`: Lengthscale parameter (divides input)
- `outer_weights`: Output scaling (typically `√(2/M)`)
- `ω`: Sampled frequencies of size `(input_dims, num_features)`
- `τ`: Sampled phases of size `(num_features,)`
- `sample_params`: Callable that returns new `(ω, τ)` samples
"""
struct RFFBasis{Tinner,Touter,Tω,Tτ,Tsample}
    inner_weights::Tinner
    outer_weights::Touter
    ω::Tω
    τ::Tτ
    sample_params::Tsample
end

(ϕ::RFFBasis)(x::KernelFunctions.ColVecs) = KernelFunctions.ColVecs(_compute_mapping(ϕ, x.X))
(ϕ::RFFBasis)(x::KernelFunctions.RowVecs) = KernelFunctions.RowVecs(_compute_mapping(ϕ, x.X')')

function _compute_mapping(ϕ::RFFBasis, X)
    X_ = X ./ ϕ.inner_weights
    ωt_x = ϕ.ω' * X_
    return ϕ.outer_weights * cos.(ωt_x .+ ϕ.τ)
end

"""
    resample!(ϕ::RFFBasis)

Resample frequencies and phases in-place.
"""
function resample!(ϕ::RFFBasis)
    ω, τ = ϕ.sample_params()
    ϕ.ω .= ω
    return ϕ.τ .= τ
end
