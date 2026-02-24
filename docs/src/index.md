```@meta
CurrentModule = GeneralizedRandomFourierFeatures
```

# GeneralizedRandomFourierFeatures.jl

A Julia package for generalized Random Fourier Features supporting a broad class of isotropic kernels.

Based on the spectral mixture representation from Langrené, Warin & Gruet (2024) "[A spectral mixture representation of isotropic kernels to generalize random Fourier features](https://arxiv.org/abs/2411.02770)".

## Overview

This package provides generalized Random Fourier Feature approximations for:

- **Generalized Cauchy Kernel**
- **Exponential Power (Generalized Gaussian) Kernel**
- **Beta Kernel**
- **Kummer Kernel**
- **Tricomi Kernel**
- **Matérn Kernel** (extended RFF sampling)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/shu13830/GeneralizedRandomFourierFeatures.jl")
```

## Quick Start

```julia
using GeneralizedRandomFourierFeatures, Random, KernelFunctions

# Create kernel and sample RFF basis
k = GeneralizedCauchyKernel(1.5, 2.0)
rng = MersenneTwister(42)
basis = sample_generalized_rff_basis(rng, k, 3, 500)

# Approximate kernel values
x, y = rand(rng, 3), rand(rng, 3)
k_approx = dot(basis(x), basis(y))
k_exact = k(x, y)
```

## API Reference

```@index
```

```@autodocs
Modules = [GeneralizedRandomFourierFeatures]
```
