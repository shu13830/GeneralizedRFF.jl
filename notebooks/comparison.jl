# =#
# # Random Fourier Features Experiments
#
# This script demonstrates three key experiments:
# 1. Comparison of true kernel matrix vs. RFF approximation
# 2. Approximation error vs. number of random features
# 3. 1D Gaussian Process Regression: true kernel vs. RFF-based approximation
#
# Convert to Jupyter notebook using Literate.jl:
# ```julia
# using Literate
# Literate.markdown("experiments.jl", "experiments.ipynb", execute=true)
# ```
# =#

using Random, LinearAlgebra
using Plots
using KernelFunctions
using RandomFourierFeatures
using GeneralizedRFFs
using Distributions
using GaussianProcesses  # for true GPR

# =#
# ## 1. Kernel Matrix Comparison
#
# We compare the exact kernel matrix computed by KernelFunctions.jl
# with the approximation obtained via Random Fourier Features.
# We'll display heatmaps side by side.
# =#

# Select test kernel and input grid
kernels = Dict(
    "Matern(ν=0.5)" => GeneralizedMaternKernel(1.0, 0.5),
    "Cauchy(β=1)"     => GeneralizedCauchyKernel(1.0, 1.0),
    "ExpPower(p=1.5)" => ExponentialPowerKernel(1.0, 1.5)
)

# Input grid
xs = range(-5.0, 5.0, length=100)

# Number of features for RFF
M = 1000
rng = MersenneTwister(42)

for (name, k) in kernels
    # True kernel matrix
    K_true = kernelmatrix(k, xs, xs)

    # RFF approximation
    φ = sample_generalized_rff_basis(rng, k, 1, M)
    F = [φ([x])[1:M] for x in xs]  # feature vectors
    Fmat = hcat(F...)
    K_rff = Fmat' * Fmat

    # Plot side by side
    p1 = heatmap(xs, xs, K_true, title="$name True Kernel", colorbar=false)
    p2 = heatmap(xs, xs, K_rff, title="$name RFF Approx.", colorbar=false)
    display(plot(p1, p2, layout=(1,2), size=(800,300)))
end

# =#
# ## 2. Approximation Error vs Number of Features
#
# We measure the relative Frobenius norm error between the true kernel matrix
# and the RFF approximation as we vary the number of random features.
# =#

feature_counts = [100, 500, 1000, 2000, 5000]

for (name, k) in kernels
    errors = Float64[]
    # Compute true kernel once
    K_true = kernelmatrix(k, xs, xs)
    norm_true = norm(K_true, fro)

    for M in feature_counts
        φ = sample_generalized_rff_basis(rng, k, 1, M)
        Fmat = hcat([φ([x])[1:M] for x in xs]...)
        K_rff = Fmat' * Fmat
        err = norm(K_rff - K_true, fro) / norm_true
        push!(errors, err)
    end

    plot(feature_counts, errors,
         label=name, xscale=:log10, yscale=:log10,
         xlabel="Number of Features", ylabel="Relative Frobenius Error",
         title="RFF Approximation Error")
end

# =#
# ## 3. 1D Gaussian Process Regression Comparison
#
# We perform Gaussian Process Regression with the true kernel and
# with an RFF-based approximation. We plot the posterior mean
# and 95% credible intervals side by side for each kernel.
# =#

# Training data
X_train = collect(-4.0:2.0:4.0)
Y_train = sin.(X_train) .+ 0.1 * randn(rng, length(X_train))

# Test inputs
X_test = range(-5.0, 5.0, length=200)

# Likelihood noise
σ = 0.1

for (name, k) in kernels
    # True GPR using GaussianProcesses.jl
    gp = GP(X_train, Y_train, MeanZero(), k, σ^2)
    post = posterior(gp, X_test)
    μ_true, var_true = mean(post), var(post)

    # RFF-based approximate GPR
    M_rff = 1000
    φ = sample_generalized_rff_basis(rng, k, 1, M_rff)
    Φ_train = hcat([φ([x])[1:M_rff] for x in X_train]...)'
    # Posterior weight distribution: Σ = (σ^-2 Φ'Φ + I)^-1
    Σ_w = inv((1/σ^2) * (Φ_train' * Φ_train) + I)
    μ_w = (1/σ^2) * Σ_w * Φ_train' * Y_train
    # Predictions
    μ_rff = [dot(φ([x])[1:M_rff], μ_w) for x in X_test]
    var_rff = [dot(φ([x])[1:M_rff], Σ_w * φ([x])[1:M_rff]) + σ^2 for x in X_test]

    # Plot results side by side
    p_true = plot(X_train, Y_train, seriestype=:scatter, label="Data",
                  title="$name True GPR", legend=:top)
    plot!(X_test, μ_true, ribbon=1.96*sqrt.(var_true), label="Mean ±2σ")

    p_rff = plot(X_train, Y_train, seriestype=:scatter, label="Data",
                 title="$name RFF GPR", legend=:top)
    plot!(X_test, μ_rff, ribbon=1.96*sqrt.(var_rff), label="Mean ±2σ")

    display(plot(p_true, p_rff, layout=(1,2), size=(800,300)))
end
