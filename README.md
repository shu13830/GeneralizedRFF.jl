# GeneralizedRFF.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://shu13830.github.io/GeneralizedRFF.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shu13830.github.io/GeneralizedRFF.jl/dev/)
[![Build Status](https://github.com/shu13830/GeneralizedRFF.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shu13830/GeneralizedRFF.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Generalized Random Fourier Features to approximate any positive definite isotropic kernels
---
**GeneralizedRFF.jl** provides Random Fourier Feature approximations for a broad class of isotropic kernels beyond the standard RBF. Leveraging spectral mixture representations from Langrené *et al.* (arXiv:2411.02770), this package supports:

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
* **Well-Tested**
  Comprehensive test suite covering kernel properties, approximation quality, and edge cases

## Installation

**Requirements:** Julia >= 1.10

```julia
julia> import Pkg
julia> Pkg.add(url = "https://github.com/shu13830/GeneralizedRFF.jl")
```

## Usage

### Quick Start: Approximate Kernel Matrix

```julia
using Random
using LinearAlgebra
using KernelFunctions
using GeneralizedRFF

# Create a Generalized Cauchy kernel
k = GeneralizedCauchyKernel(1.5, 2.0)

# Generate random data points
rng = MersenneTwister(42)
X = [rand(rng, 3) for _ in 1:50]  # 50 points in 3D

# Compute exact kernel matrix
K_exact = kernelmatrix(k, X)

# Compute RFF approximation
K_approx = rff_kernelmatrix(rng, k, X, 500)  # 500 random features

# Compare
println("Relative error: ", norm(K_exact - K_approx) / norm(K_exact))
```

## Approximate Gaussian Processes with Generalized RFFs

```julia
using Random
using AbstractGPs
using KernelFunctions: ColVecs
using GeneralizedRFF

# Create a GP with a generalized kernel
k = KummerKernel(α=1.5, β=2.0, γ=1.5)
f = GP(k)

# Build RFF weight-space approximation
rng = MersenneTwister(123)
approx = GeneralizedRFF.build_grff_weight_space_approx(rng, 3, 200)  # 3D input, 200 features
f_approx = approx(f)

# Use for inference with training data (wrapped in ColVecs)
X_train = ColVecs(randn(3, 20))  # 20 points in 3D
y_train = sin.(X_train.X[1, :]) + 0.1 * randn(20)
posterior_approx = posterior(f_approx(X_train, 0.1), y_train)
```

## API Reference

### Main Functions

* **`sample_generalized_rff_basis(rng, kernel, input_dims, num_features)`**
  Sample a Random Fourier Feature basis for any supported kernel.
  - Returns: `RFFBasis` object representing the random feature map

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

* **`ExponentialPowerKernel(α)` / `GeneralizedGaussianKernel(α)` / `SubbotinKernel(α)`**
  Exponential power kernel: k(r) = exp(-r^α), α ∈ (0, 2]

* **`MaternKernel(ν)`** *(from KernelFunctions.jl)*
  Standard Matérn kernel (supported for generalized RFF sampling)

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
