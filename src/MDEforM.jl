module MDEforM

# required packages
using CairoMakie
using DifferentialEquations
using HCubature
using LaTeXStrings
using Optim
using Dates
using LinearAlgebra
using NaNMath
using QuadGK
using DSP
using Interpolations

# main code
export Fast_OU, LDA, NLDAM, NSDP
export Langevin, K, LDO, NLDO
export Burger
export Fast_chaotic
export produce_trajectory

include("multiscale_limit_pairs.jl")

export μ, ∂ϑ_μ, ∂Σ_μ

include("invariant_densities.jl")

export Δ, k

include("MDE_functionals.jl")

export Δ_grad_ϑ, Δ_grad_Σ

include("MDE_gradients.jl")

export Σ_∞_QdP
export Σ_∞_QrP

include("MDE_asymptotic_variances.jl")

export MDE

include("MDE_optimizers.jl")

end
