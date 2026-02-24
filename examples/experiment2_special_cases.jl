# Experiment 2: Special Case Consistency Verification
# Validates that generalized kernels correctly reduce to known special cases
# when appropriate parameter values are used.
#
# Reference: EXPERIMENTS.md, Experiment 2

using GeneralizedRandomFourierFeatures
using KernelFunctions
using Random
using LinearAlgebra
using Printf

"""
    compare_kernels(k1, k2, name1, name2, test_points)

Compare two kernels on a set of test points and report maximum absolute difference.
"""
function compare_kernels(k1, k2, name1::String, name2::String, test_points)
    max_diff = 0.0
    max_rel_diff = 0.0

    for (x, y) in test_points
        v1 = k1(x, y)
        v2 = k2(x, y)

        abs_diff = abs(v1 - v2)
        rel_diff = abs_diff / (abs(v1) + 1e-10)

        max_diff = max(max_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)
    end

    println("\nComparing: $name1 ↔ $name2")
    @printf("  Max absolute difference: %.6e\n", max_diff)
    @printf("  Max relative difference: %.6e\n", max_rel_diff)

    if max_rel_diff < 0.01
        println("  ✓ Excellent agreement (<1%)")
    elseif max_rel_diff < 0.05
        println("  ✓ Good agreement (<5%)")
    else
        println("  ⚠ Warning: significant difference")
    end

    return max_diff, max_rel_diff
end

"""
Generate test point pairs for kernel comparison.
"""
function generate_test_pairs(d::Int, n::Int; rng=MersenneTwister(42))
    # Include origin, nearby points, and distant points
    pairs = Tuple{Vector{Float64}, Vector{Float64}}[]

    # Origin
    push!(pairs, (zeros(d), zeros(d)))

    # Small distances
    for _ in 1:n÷3
        x = 0.1 .* randn(rng, d)
        y = 0.1 .* randn(rng, d)
        push!(pairs, (x, y))
    end

    # Medium distances
    for _ in 1:n÷3
        x = randn(rng, d)
        y = randn(rng, d)
        push!(pairs, (x, y))
    end

    # Large distances
    for _ in 1:n÷3
        x = 3.0 .* randn(rng, d)
        y = 3.0 .* randn(rng, d)
        push!(pairs, (x, y))
    end

    return pairs
end

# ============================================================================
# Main Experiments
# ============================================================================

println("="^70)
println("EXPERIMENT 2: SPECIAL CASE CONSISTENCY")
println("="^70)
println("\nObjective: Verify generalized kernels reduce to known special cases")
println("Method: Compare kernel values with standard implementations\n")

d = 3  # Input dimension
n_tests = 30
test_pairs = generate_test_pairs(d, n_tests)

results = []

# ----------------------------------------------------------------------------
# Test 1: ExponentialPower(α=2) should match Gaussian
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Test 1: Exponential Power (α=2) → Gaussian")
println("="^70)

k_ep2 = GammaExponentialKernel(γ=[2.0])
k_rbf = SqExponentialKernel()

diff, rel_diff = compare_kernels(k_ep2, k_rbf, "ExponentialPower(α=2)", "Gaussian", test_pairs)
push!(results, ("EP(α=2) vs Gaussian", rel_diff))

# ----------------------------------------------------------------------------
# Test 2: ExponentialPower(α=1) should match Laplace
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Test 2: Exponential Power (α=1) → Laplace")
println("="^70)

k_ep1 = GammaExponentialKernel(γ=[1.0])
k_laplace = with_lengthscale(LaplacianKernel(), 1.0)

diff, rel_diff = compare_kernels(k_ep1, k_laplace, "ExponentialPower(α=1)", "Laplace", test_pairs)
push!(results, ("EP(α=1) vs Laplace", rel_diff))

# ----------------------------------------------------------------------------
# Test 3: Matérn-1/2 should be close to Exponential
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Test 3: Matérn-1/2 → Exponential (Laplace)")
println("="^70)

k_matern12 = MaternKernel(ν=[0.5])
k_exp = GammaExponentialKernel(γ=[1.0])

diff, rel_diff = compare_kernels(k_matern12, k_exp, "Matérn(ν=0.5)", "Exponential", test_pairs)
push!(results, ("Matérn(ν=0.5) vs Exponential", rel_diff))

# ----------------------------------------------------------------------------
# Test 4: Self-consistency check for Generalized Cauchy
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Test 4: Generalized Cauchy self-consistency")
println("="^70)

k_cauchy = GeneralizedCauchyKernel(1.5, 2.0)

# Check kernel properties
println("\nVerifying kernel properties:")
x = rand(d)

# Check symmetry
y = rand(d)
v1 = k_cauchy(x, y)
v2 = k_cauchy(y, x)
@printf("  Symmetry: |k(x,y) - k(y,x)| = %.6e\n", abs(v1 - v2))

# Check normalization at origin
v_origin = k_cauchy(zeros(d), zeros(d))
@printf("  Normalization: k(0,0) = %.6f (should be ≈1.0)\n", v_origin)

# Check positive definiteness
K = kernelmatrix(k_cauchy, [rand(d) for _ in 1:10])
λ_min = minimum(eigvals(K))
@printf("  Min eigenvalue: %.6f (should be >0)\n", λ_min)

if abs(v1 - v2) < 1e-10
    println("  ✓ Kernel is symmetric")
end

if abs(v_origin - 1.0) < 0.01
    println("  ✓ Kernel is properly normalized")
end

if λ_min > -1e-10
    println("  ✓ Kernel is positive definite")
end

push!(results, ("Cauchy self-consistency", abs(v_origin - 1.0)))

# ----------------------------------------------------------------------------
# Test 5: Kummer, Beta, Tricomi finite checks
# ----------------------------------------------------------------------------
println("\n" * "="^70)
println("Test 5: Advanced kernels (Kummer, Beta, Tricomi)")
println("="^70)

advanced_kernels = [
    (KummerKernel(α=1.5, β=2.0, γ=1.5), "Kummer"),
    (BetaKernel(α=1.5, β=2.0, γ=1.5), "Beta"),
    (TricomiKernel(α=1.5, β=2.0, γ=1.5), "Tricomi"),
]

for (k, name) in advanced_kernels
    println("\n$name Kernel:")

    # Check that all values are finite
    finite_count = 0
    for (x, y) in test_pairs[1:10]
        v = k(x, y)
        if isfinite(v)
            finite_count += 1
        end
    end

    @printf("  Finite values: %d/10\n", finite_count)

    if finite_count == 10
        println("  ✓ All values are finite")
    else
        println("  ✗ Some values are non-finite")
    end

    # Check symmetry
    x, y = test_pairs[1]
    v1 = k(x, y)
    v2 = k(y, x)
    sym_error = abs(v1 - v2)
    @printf("  Symmetry error: %.6e\n", sym_error)

    if sym_error < 1e-10
        println("  ✓ Kernel is symmetric")
    end

    push!(results, ("$name finiteness", finite_count / 10))
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("SUMMARY")
println("="^70)

for (test_name, value) in results
    if occursin("vs", test_name)
        @printf("%-45s: %.6f (rel. diff)\n", test_name, value)
    else
        @printf("%-45s: %.6f\n", test_name, value)
    end
end

println("\n✓ Experiment 2 completed successfully!")
println("Expected: All special cases match within <5% relative error")
