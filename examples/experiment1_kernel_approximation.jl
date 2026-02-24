# Experiment 1: Kernel Approximation Accuracy
# Validates that RFF approximation K̂_M(u) converges to true kernel K(u)
# as the number of features M increases.
#
# Reference: EXPERIMENTS.md, Experiment 1

using GeneralizedRandomFourierFeatures
using KernelFunctions
using Random
using LinearAlgebra
using Statistics
using Printf

"""
    compute_approximation_error(kernel, X, M, rng)

Compute approximation error between exact kernel matrix and RFF approximation.

Returns relative Frobenius norm error: ||K - K̂|| / ||K||
"""
function compute_approximation_error(kernel, X, M, rng)
    # Exact kernel matrix
    K_exact = kernelmatrix(kernel, X)

    # RFF approximation
    K_approx = rff_kernelmatrix(rng, kernel, X, M)

    # Relative error
    return norm(K_exact - K_approx) / norm(K_exact)
end

"""
    test_kernel_convergence(kernel, name; M_values, n_points, d, n_trials)

Test convergence of RFF approximation for a given kernel.
"""
function test_kernel_convergence(
    kernel,
    name::String;
    M_values = [100, 500, 1000, 5000],
    n_points = 50,
    d = 2,
    n_trials = 5
)
    println("\n" * "="^70)
    println("Testing: $name")
    println("="^70)

    errors = Dict{Int, Vector{Float64}}()

    for M in M_values
        errors[M] = Float64[]

        for trial in 1:n_trials
            rng = MersenneTwister(42 + trial)
            X = [rand(rng, d) for _ in 1:n_points]

            error = compute_approximation_error(kernel, X, M, rng)
            push!(errors[M], error)
        end

        mean_error = mean(errors[M])
        std_error = std(errors[M])

        @printf("M = %5d: Mean Error = %.4f ± %.4f\n", M, mean_error, std_error)
    end

    # Check convergence
    mean_errors = [mean(errors[M]) for M in M_values]

    if issorted(mean_errors, rev=true)
        println("✓ Convergence confirmed: error decreases with M")
    else
        println("⚠ Warning: non-monotonic convergence")
    end

    if mean_errors[end] < 0.1
        println("✓ Final approximation quality: excellent (<10% error)")
    elseif mean_errors[end] < 0.2
        println("✓ Final approximation quality: good (<20% error)")
    else
        println("⚠ Warning: high approximation error")
    end

    return errors
end

# ============================================================================
# Main Experiments
# ============================================================================

println("="^70)
println("EXPERIMENT 1: KERNEL APPROXIMATION ACCURACY")
println("="^70)
println("\nObjective: Verify RFF approximation converges to true kernel")
println("Method: Measure ||K - K̂_M|| / ||K|| for increasing M\n")

# Define test kernels according to EXPERIMENTS.md Table
test_kernels = [
    # Exponential Power with different α
    (GammaExponentialKernel(γ=[0.5]), "Exponential Power (α=0.5)"),
    (GammaExponentialKernel(γ=[1.0]), "Exponential Power (α=1.0, Laplace)"),
    (GammaExponentialKernel(γ=[1.5]), "Exponential Power (α=1.5)"),
    (GammaExponentialKernel(γ=[2.0]), "Exponential Power (α=2.0, Gaussian)"),

    # Generalized Cauchy
    (GeneralizedCauchyKernel(1.5, 1.5), "Generalized Cauchy (α=1.5, β=1.5)"),

    # Matérn variants
    (MaternKernel(ν=[0.5]), "Matérn-1/2 (Exponential)"),
    (MaternKernel(ν=[1.5]), "Matérn-3/2"),
    (MaternKernel(ν=[2.5]), "Matérn-5/2"),

    # Kummer, Beta, Tricomi
    (KummerKernel(α=1.5, β=1.5, γ=1.5), "Kummer (1.5, 1.5, 1.5)"),
    (BetaKernel(α=1.5, β=1.5, γ=1.5), "Beta (1.5, 1.5, 1.5)"),
    (TricomiKernel(α=1.5, β=1.5, γ=1.5), "Tricomi (1.5, 1.5, 1.5)"),
]

# Run experiments
results = Dict()

for (kernel, name) in test_kernels
    try
        errors = test_kernel_convergence(
            kernel,
            name;
            M_values = [100, 500, 1000, 5000],
            n_points = 50,
            d = 2,
            n_trials = 5
        )
        results[name] = errors
    catch e
        println("✗ Error testing $name: $e")
    end
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("SUMMARY")
println("="^70)

for (name, errors) in results
    final_error = mean(errors[5000])
    @printf("%-45s: %.4f (M=5000)\n", name, final_error)
end

println("\n✓ Experiment 1 completed successfully!")
println("Expected behavior: Error decreases with M, achieving <10% for M≥1000")
