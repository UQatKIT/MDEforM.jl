# just a scratch file for repeated testing during development; instead of using the REPL for large code chunks

using Revise    # for continuous testing without restarting Julia after each change to the main code
using LinearAlgebra
using MDEforM

struct BilinearFastSlow{V<:AbstractVector{<:Real}, M<:AbstractMatrix{<:Real}, R<:Real}
    initial_data::V
    γ::V
    D::M
    σ::M
    α::R 
    Σ::R 
    c::R
end

# general multiscale system
function simulate(model::BilinearFastSlow; ε::Real=0.1, T::Real=100, dt::Real=1e-3)
    N = convert(Int64, T/dt)
    m = length(model.initial_data)-1

    X = Vector{Float64}(undef, N)
    Y = Array{Float64}(undef, m, N)
    X[1] = model.initial_data[1]
    Y[:,1] = model.initial_data[2:end]

    for i in 1:(N-1)
        dW_x = sqrt(dt)*randn(1)[1]
        dW_y = sqrt(dt)*randn(m)

        X[i+1] = X[i] + (1/ε*(model.γ ⋅ Y[:,i]) - model.α*X[i])dt + model.Σ*dW_x
        Y[:,i+1] = Y[:,i] + (-1/ε^2*model.D*Y[:,i] - model.c/ε*X[i]*Y[:,i])dt + 1/ε*model.σ*dW_y
    end

    (X, Y)
end

xy0 = [1.0, 2.0, 3.0, 5.0]  # initial data
γ = [1.0, -2.0, 1.0]      # 3-dimensional vector
D = [2.5 1.0 0.0;   # Hurwitz matrix
     0.0 2.0 0.8;
     0.0 0.0 1.5]
σ = [1.0 0.3 0.0;   # σ*σ^⊤ must be positive definite
     0.2 0.9 0.4;
     0.0 0.1 0.8]
α = 1.0             # must be positive
Σ = 0.5             # must be positive
c = -1.0            # must be real

model = BilinearFastSlow(xy0, γ, D, σ, α, Σ, c)

XY_system = simulate(model, ε=0.1, T=10.0, dt=1e-3)

MDEforM.produce_trajectory(XY_system[1], 10)
MDEforM.produce_trajectory(XY_system[2][3,:], 10)

XY_system[2]