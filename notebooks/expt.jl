# RFF Approximation Error by Kernel
#
# This notebook demonstrates how to compute and visualize the approximation error
# of Random Fourier Features (RFF) for various isotropic kernels provided by
# the GeneralizedRFFs package.
#
# To convert this script into a Jupyter notebook, run:
#
#     using Literate
#     Literate.markdown("experiment_generalized_rff_literate.jl", "notebook.ipynb")

# ## 1. Load Required Packages
using Random, LinearAlgebra
using KernelFunctions
using RandomFourierFeatures
using GeneralizedRFFs           ## Provides sample_grff_basis and custom kernels
using Plots

# ## 2. Data Preparation
Random.seed!(1234)
input_dim = 2              ## Number of input dimensions
num_samples = 200          ## Number of data points
X = randn(input_dim, num_samples)  ## Each column is a sample vector

# ## 3. Define Kernels to Test
kernels = [
    SqExponentialKernel(),                   ## RBF kernel
    MaternKernel(3/2),                       ## Matérn kernel (ν = 3/2)
    SubbotinKernel(γ=1.5),                   ## Exponential Power kernel (γ = 1.5)
    GeneralizedCauchyKernel(α=1.2, β=2.0),    ## Generalized Cauchy kernel
    BetaKernel(α=1.5, β=2.0, γ=3.0),          ## Beta kernel
    KummerKernel(α=1.5, β=2.0, γ=3.0),        ## Kummer kernel
    TricomiKernel(α=1.5, β=2.0, γ=3.0)        ## Tricomi kernel
]
kernel_names = [
    "RBF", "Matérn3/2", "ExpPower(1.5)", "GenCauchy(1.2,2)",
    "Beta(1.5,2,3)", "Kummer(1.5,2,3)", "Tricomi(1.5,2,3)"
]

# ## 4. Compute Exact Kernel Matrices
K_exact = Dict{String, Matrix{Float64}}()
for (kernel, name) in zip(kernels, kernel_names)
    K_exact[name] = kernelmatrix(kernel, eachcol(X), eachcol(X))
end

# ## 5. Evaluate RFF Approximation Error
feature_counts = [100, 500, 1000, 2000]
errors = Dict(name => Float64[] for name in kernel_names)

for (kernel, name) in zip(kernels, kernel_names)
    for M in feature_counts
        ## Sample an RFF basis and transform data
        basis = sample_grff_basis(MersenneTwister(42), kernel, input_dim, M)
        Z = transform(basis, X')  ## Z is M × num_samples
        ## Approximate kernel matrix via inner products
        K_approx = Z' * Z
        ## Compute relative Frobenius-norm error
        err = norm(K_approx - K_exact[name], fro) / norm(K_exact[name], fro)
        push!(errors[name], err)
    end
end

# ## 6. Visualization
plot(
    feature_counts,
    errors[kernel_names[1]],
    xlabel = "Number of Random Features (M)",
    ylabel = "Relative Error (Frobenius norm)",
    xscale = :log10,
    yscale = :log10,
    title  = "RFF Approximation Error by Kernel",
    legend = :outertopright,
    lw     = 2,
    marker = :auto,
)
for name in kernel_names[2:end]
    plot!(feature_counts, errors[name], label = name, lw = 2, marker = :auto)
end

display(current())
