# GeneralizedRFF.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://shu13830.github.io/GeneralizedRFF.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shu13830.github.io/GeneralizedRFF.jl/dev/)
[![Build Status](https://github.com/shu13830/GeneralizedRFF.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shu13830/GeneralizedRFF.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Generalized Random Fourier Features to approximate any positive definite isotropic kernels
---
**GeneralizedRFF.jl** extends [RandomFourierFeatures.jl](https://github.com/JuliaGaussianProcesses/RandomFourierFeatures.jl) by providing Random Fourier Feature approximations for a broad class of isotropic kernels beyond the standard RBF. Leveraging spectral mixture representations from Langrené *et al.* (arXiv:2411.02770), this package supports:

* **Generalized Matérn Kernel**
* **Generalized Cauchy Kernel**
* **Exponential Power (Generalized Gaussian) Kernel**
* **Beta Kernel**
* **Kummer Kernel**
* **Tricomi Kernel**

## Features

* **Unified Spectral Framework**
  All kernels use the spectral mixture representation from Theorem 1 of Langrené et al. (2024): η = (λR)^(1/α) · S_α

* **Broad Kernel Support**
  Six families of isotropic kernels beyond the standard RBF, including heavy-tailed and hypergeometric kernels

* **Numerical Stability**
  - Optimized α = 2 (Gaussian) special case
  - Log-space computations for Beta and Tricomi kernels
  - Robust symmetric α-stable sampling

* **Automatic Differentiation Compatible**
  - Parameters stored as `Vector{T}` for AD compatibility
  - Full `Functors.jl` integration for gradient-based optimization

* **Seamless Integration**
  - Works with `KernelFunctions.jl` ecosystem
  - Compatible with `AbstractGPs.jl` for Gaussian process inference
  - Extends `RandomFourierFeatures.jl`

* **Well-Tested**
  Comprehensive test suite covering kernel properties, approximation quality, and edge cases

## Installation

```julia
julia> import Pkg
julia> Pkg.add(url = "https://github.com/shu13830/GeneralizedRFF.jl")
```

## Usage

### Quick Start: Approximate Kernel Values

```julia
using Random
using KernelFunctions
using GeneralizedRFF

# Create a Generalized Cauchy kernel
k = GeneralizedCauchyKernel(1.5, 2.0)

# Sample RFF basis
rng = MersenneTwister(42)
basis = sample_generalized_rff_basis(rng, k, 3, 500)  # 3D input, 500 features

# Map data points to feature space
x = rand(rng, 3)
y = rand(rng, 3)
φx = basis(x)
φy = basis(y)

# Approximate kernel value
k_approx = dot(φx, φy)
k_exact = k(x, y)

println("Exact: $k_exact, Approx: $k_approx")
```

### Approximate Kernel Matrices

```julia
using LinearAlgebra

# Generate data points
X = [rand(rng, 3) for _ in 1:50]

# Compute exact kernel matrix
K_exact = kernelmatrix(k, X)

# Compute RFF approximation
K_approx = rff_kernelmatrix(rng, k, X, 500)

# Compare
println("Relative error: ", norm(K_exact - K_approx) / norm(K_exact))
```

## Approximate Gaussian Processes with Generalized RFFs

```julia
using AbstractGPs
using BayesianLinearRegressors
using GeneralizedRFF

# Create a GP with a generalized kernel
k = KummerKernel(α=1.5, β=2.0, γ=1.5)
f = GP(k)

# Build RFF weight-space approximation
rng = MersenneTwister(123)
approx = build_grff_weight_space_approx(rng, 3, 200)  # 3D input, 200 features
f_approx = approx(f)

# Use for inference...
```

## API Reference

### Main Functions

* **`sample_generalized_rff_basis(rng, kernel, input_dims, num_features)`**
  Sample a Random Fourier Feature basis for any supported kernel.
  - Returns: `RFFBasis` object compatible with `RandomFourierFeatures.jl`

* **`rff_kernelmatrix(rng, kernel, X, num_features)`**
  Compute approximate kernel matrix using RFF.
  - `X`: Vector of data points
  - Returns: Approximate kernel matrix (N × N)

* **`rff_kernelmatrix(basis, X)`**
  Compute approximate kernel matrix from pre-sampled basis.

### Supported Kernels

All kernels are subtypes of `KernelFunctions.Kernel`:

* **`GeneralizedCauchyKernel(α, β)`**
  Generalized Cauchy kernel with exponent `α ∈ (0,2]` and tail parameter `β > 0`

* **`KummerKernel(; α, β, γ)`**
  Kummer confluent hypergeometric kernel

* **`BetaKernel(; α, β, γ)`**
  Beta kernel with spectral Beta mixture

* **`TricomiKernel(; α, β, γ)`**
  Tricomi confluent hypergeometric kernel

* **`GammaExponentialKernel(γ)` / `ExponentialPowerKernel` / `SubbotinKernel`**
  Exponential power kernel: k(r) = exp(-r^γ)

* **`MaternKernel(ν)`** *(from KernelFunctions.jl)*
  Standard Matérn kernel (extended for generalized RFF sampling)

### Integration with Functors.jl

All kernels are compatible with `Functors.jl` for automatic differentiation and parameter optimization:

```julia
using Functors

k = GeneralizedCauchyKernel(1.5, 2.0)
params = Functors.fmap(identity, k)  # Extract parameters
```

## Testing

Run the built‑in test suite:

```shell
julia> import Pkg; Pkg.test("GeneralizedRFF")
```

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or pull request on the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
