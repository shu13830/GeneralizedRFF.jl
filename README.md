# GeneralizedRFFs.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://shu13830.github.io/GeneralizedRFFs.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shu13830.github.io/GeneralizedRFFs.jl/dev/)
[![Build Status](https://github.com/shu13830/GeneralizedRFFs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shu13830/GeneralizedRFFs.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Generalized Random Fourier Features to approximate any positive definite isotropic kernels
---
**GeneralizedRFFs.jl** extends [RandomFourierFeatures.jl](https://github.com/example/RandomFourierFeatures.jl) by providing Random Fourier Feature approximations for a broad class of isotropic kernels beyond the standard RBF. Leveraging spectral mixture representations from Langrené *et al.* (arXiv:2411.02770), this package supports:

* **Generalized Matérn Kernel**
* **Generalized Cauchy Kernel**
* **Exponential Power (Generalized Gaussian) Kernel**
* **Beta Kernel**
* **Kummer Kernel**
* **Tricomi Kernel**

## Features

* **One-line integration**<br>Sample feature maps for both RBF and extended kernels via a single API function.
* **Spectral mixture sampling**<br>Automatic sampling of scale mixtures and stable distributions for accurate kernel approximations.
* **Compatibility**<br>Works seamlessly with `KernelFunctions.jl` types and the existing RFF infrastructure in `RandomFourierFeatures.jl`.
* **Lightweight**<br>Minimal dependencies: `Distributions.jl`, `SpecialFunctions.jl`, `RandomFourierFeatures.jl`, `KernelFunctions.jl`.

## Installation

```julia
julia> import Pkg
julia> Pkg.add(url = "https://github.com/shu13830/GeneralizedRFFs.jl")
```

## Usage
### Approximate kernel
```julia
using Random: MersenneTwister
using KernelFunctions
using Plots
using GeneralizedRFFs

# Instantiate an extended kernel
k = GeneralizedMaternKernel(ℓ=1.0, ν=0.5)

# Prepare RNG and sample RFF basis
rng = MersenneTwister(42)
φ = sample_generalized_rff_basis(rng, k, input_dims=3, num_features=500)

# Map data points x, y ∈ ℝ³ to feature space
x = rand(3); y = rand(3)
kx = φ(x)
ky = φ(y)


# Approximate kernel matrix by inner product
K_approx = dot(kx, ky)

# compute kernel matrix
K = kernelmatrix(k, x, y)

# compare kernel matrices
plot(heatmap(K_approx), heatmap(K))

```

## Approximate Gaussian processes with generalized RFFs
```julia
using Random: MersenneTwister
using KernelFunctions
using Plots
using AbstractGPs
using BayesianLinearRegressors
using GeneralizedRFFs

# Instantiate an extended kernel
k = GeneralizedMaternKernel(ℓ=1.0, ν=0.5)

# Prepare RNG and sample RFF basis
rng = MersenneTwister(42)
φ = sample_generalized_rff_basis(rng, k, input_dims=3, num_features=500)

# ...

```

## API Reference

* `sample_generalized_rff_basis(rng::AbstractRNG, k::Kernel, d::Int, M::Int)`<br>
  Returns an `RFFBasis` for kernel `k` on `d`‑dimensional inputs with `M` random features.

* **Kernel types** (all subtype `KernelFunctions.Kernel`):

  * `GeneralizedMaternKernel(ℓ::Float64, ν::Float64)`
  * `GeneralizedCauchyKernel(ℓ::Float64, β::Float64)`
  * `ExponentialPowerKernel(ℓ::Float64, p::Float64)`
  * `BetaKernel(ℓ::Float64, α::Float64, β::Float64)`
  * `KummerKernel(ℓ::Float64, α::Float64, β::Float64)`
  * `TricomiKernel(ℓ::Float64, α::Float64, β::Float64)`

## Testing

Run the built‑in test suite:

```shell
julia> import Pkg; Pkg.test("GeneralizedRFFs")
```

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or pull request on the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
