########################################################################################################
########################################################################################################
# Simulation of different multiscale SDE systems and their respective SDE limits obtained through
# homogenization theory. Considered classes of processes involve fast Ornstein-Uhlenbeck, overdamped
# Langevin equation with a large-scale potential and fast oscillating part in 1D and 2D, truncated 
# Burger's equation and a deterministic fast chaotic noise system.
########################################################################################################
########################################################################################################
# Jaroslav Borodavka, 08.08.2024

########################################################################################################
## functions for the realization of multiscale and limit processes, different examples
########################################################################################################

## Fast Ornstein-Uhlenbeck ##

# general mutliscale system for fast Ornstein-Uhlenbeck process
# @doc raw""" """ is a command combo to avoid escaping latex commands as \\epsilon
@doc raw"""
    Fast_OU(x0::Real, y0::Real; <keyword arguments>)

Return a 2-dimensional fast-slow Ornstein-Uhlenbeck process starting at `(x0, y0)` as a discretized time series.

The corresponding stochastic differential equation is defined for ``t \in [0,T]`` as
```math
\begin{aligned}
  dX_ϵ(t) &= \left( \frac{1}{ϵ} σ(X_ϵ(t)) Y_ϵ(t) + h(X_ϵ(t), Y_ϵ(t)) - σ'(X_ϵ(t))σ(X_ϵ(t)) \right) dt, \quad &X_ϵ(0) = x_0, \\
  dY_ϵ(t) &= -\frac{1}{ϵ^2} Y_ϵ(t) + \frac{\sqrt{2}}{ϵ} dV(t), \quad &Y_ϵ(0) = y_0.
\end{aligned}
```
Here, ``σ'`` is the first derivative of ``σ``. A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `x0::Real`:                             initial point ``x_0`` of slow process ``X_ϵ``.
- `y0::Real`:                             initial point ``y_0`` of fast process ``Y_ϵ``.
- `func_config`:                          collection of the functions ``h, σ`` and ``σ'``.
- `ϵ::Real=0.1`:                          positive small scale parameter ``ϵ``.
- `T::Real=100`:                          time horizon of time series.
- `dt::Real=1e-3`:                        time discretization step used in the Euler-Maruyama scheme.

---
# Examples
```julia
# linear drift with additive noise
T = 10.0
trajectory = Fast_OU(1.0, 1.0, func_config=LDA(), ϵ=0.1, T=T)
fig = produce_trajectory(trajectory, T)
#save("trajectory_fast_OU_process.pdf", fig)

# nonlinear drift with additive and multiplicative noise
T = 10.0
trajectory = Fast_OU(1.0, 1.0, func_config=NLDAM(), ϵ=0.1, T=T)
fig = produce_trajectory(trajectory, T)

# nonlinear drift with additive and multiplicative noise, non-symmetric double-well potential
T = 10.0
trajectory = Fast_OU(1.0, 1.0, func_config=NSDP(), ϵ=0.1, T=T)
fig = produce_trajectory(trajectory, T)
```

---
See also [`LDA`](@ref), [`NLDAM`](@ref), [`NSDP`](@ref).
"""
function Fast_OU(x0::Real, y0::Real; func_config, ϵ::Real=0.1, T::Real=100, dt::Real=1e-3)
    h = func_config[1]
    σ = func_config[2]
    σ_prime = func_config[3]

    N = convert(Int64, T/dt) - 1
  
    X = Vector{Float64}(undef, N+1)
    Y = Vector{Float64}(undef, N+1)
    X[1] = x0
    Y[1] = y0
    
    for k in 1:N
      dW = sqrt(dt)*randn(1)[1]
      X[k+1] = X[k] + (1/ϵ*σ(X[k])Y[k] + h(X[k]) - σ_prime(X[k])σ(X[k]))dt
      Y[k+1] = Y[k] + (-1/ϵ^2*Y[k])dt + sqrt(2)/ϵ*dW
    end
    
    (X, Y)
end

# general limit process for fast Ornstein-Uhlenbeck process
@doc raw"""
    Fast_OU(X0::Real; <keyword arguments>)

Return a one-dimensional limit process, homogenized from the fast-slow Ornstein-Uhlenbeck process, starting at `X0` as a discretized time series.

The corresponding stochastic differential equation is defined for ``t \in [0,T]`` as
```math
\begin{aligned}
  dX(t) = \bar{h}(X(t)) dt - \sqrt{2 σ(X(t))^2} dW(t), \quad X(0) = X_0.
\end{aligned}
```
Here, ``\bar{h}`` is the average of ``h`` with respect to the invariant measure of the fast process ``Y_ϵ`` of the fast-slow Ornstein-Uhlenbeck process. 
A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `X0::Real`:                               initial point ``X_0`` of limit process ``X``.
- `func_config`:                            collection of the functions ``\bar{h}`` and ``σ``.
- `T::Real=100`:                            time horizon of time series.
- `dt::Real=1e-3`:                          time discretization step used in the Euler-Maruyama scheme.

---
# Examples
```julia-repl
julia> h_aver = x -> -x   # corresponds to an ordinary Ornstein-Uhlenbeck process
julia> σ = x -> sqrt(2)
julia> Fast_OU(1.0, func_config=(h_aver, σ))
```

---
See also [`LDA`](@ref), [`NLDAM`](@ref), [`NSDP`](@ref).
"""
function Fast_OU(X0::Real; func_config, T::Real=100, dt::Real=1e-3)
  h_aver = func_config[1]
  σ = func_config[2]

  N = convert(Int64, T/dt) - 1

  X = Vector{Float64}(undef, N+1)
  X[1] = X0
  
  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] + h_aver(X[k])dt + sqrt(σ(X[k])^2)dW
  end
  
  X
end

# linear drift with additive noise
@doc raw"""
    LDA(A::Real=1, σ::Real=1)

Return a tuple of functions used for the definition of [`Fast_OU`](@ref).

The returned functions are
```math
\begin{aligned}
  &h(x) = -Ax, \quad &A > 0, \\
  &σ(x) = \sqrt{σ},\quad
  σ'(x) = 0,  \quad &σ>0.
\end{aligned}
```
They yield a linear drift with additive noise.

---
# Arguments
- `A::Real=1`:        non-negative real number.
- `σ::Real=1`:        positive real number.

---
See also [`Fast_OU`](@ref).
"""
function LDA(A::Real=1, σ::Real=1)
  (x -> -A*x, x -> sqrt(σ), x -> 0)
end

# nonlinear drift with additive and multiplicative noise
@doc raw"""
    NLDAM(A::Real=2, B::Real=10, σ_a::Real=1, σ_b::Real=1)

Return a tuple of functions used for the definition of [`Fast_OU`](@ref).

The returned functions are
```math
\begin{aligned}
  &h(x) = Ax - Bx^3, \quad &A, B > 0, \\
  &σ(x) = \sqrt{σ_a + σ_b x^2}, \quad
  σ'(x) = \frac{σ_b x}{\sqrt{σ_a + σ_b x^2}}, \quad &σ_a, σ_b > 0.
\end{aligned}
```
They yield a nonlinear drift with additive and multiplicative noise.

---
# Arguments
- `A::Real=2`:          non-negative real number.
- `B::Real=10`:         non-negative real number.
- `σ_a::Real=1`:        positive real number.
- `σ_b::Real=1`:        positive real number.

---
See also [`Fast_OU`](@ref).
"""
function NLDAM(A::Real=2, B::Real=10, σ_a::Real=1, σ_b::Real=1)
  h = x -> A*x - B*x^3
  σ = x -> sqrt(σ_a + σ_b*x^2)
  σ_prime = x -> σ_b*x/sqrt(σ_a + σ_b*x^2)

  (h, σ, σ_prime)
end
  
# nonlinear drift with additive and multiplicative noise, non-symmetric double-well potential
@doc raw"""
    NSDP(A::Real=1, B::Real=2, C::Real=5, σ_a::Real=1, σ_b::Real=1)

Return a tuple of functions used for the definition of [`Fast_OU`](@ref).

The returned functions are
```math
\begin{aligned}
  &h(x) = Ax + Bx^2 - Cx^3, \quad &A, B, C > 0, \\
  &σ(x) = \sqrt{σ_a + σ_b x^2}, \quad
  σ'(x) = \frac{σ_b x}{\sqrt{σ_a + σ_b x^2}}, \quad &σ_a, σ_b > 0.
\end{aligned}
```
They yield a nonlinear, non-symmetric double-well potential drift with additive and multiplicative noise.

---
# Arguments
- `A::Real=1`:          non-negative real number.
- `B::Real=2`:          non-negative real number.
- `C::Real=5`:          non-negative real number.
- `σ_a::Real=1`:        positive real number.
- `σ_b::Real=1`:        positive real number.

---
See also [`Fast_OU`](@ref).
"""
function NSDP(A::Real=1, B::Real=2, C::Real=5, σ_a::Real=1, σ_b::Real=1)
  h = x -> A*x + B*x^2 - C*x^3
  sigma = x -> sqrt(σ_a + σ_b*x^2)
  sigma_prime = x -> σ_b*x/sqrt(σ_a + σ_b*x^2)

  (h, sigma, sigma_prime)
end

## Overdamped Langevin process with large-scale potential and fast oscillating part ##

# general multiscale system for overdamped Langevin process with large-scale potential V and fast oscillating part p
@doc raw"""
    Langevin(x0::Real, y0::Real; <keyword arguments>)

Return a 2-dimensional overdamped Langevin process with a large-scale potential and a fast oscillating part starting at `(x0, y0)` as a discretized time series.

The corresponding stochastic differential equation is defined for ``t \in [0,T]`` as
```math
\begin{aligned}
  dX_ϵ(t) = -α V'(X_ϵ(t)) - \frac{1}{ϵ} p'\left( \frac{X_ϵ(t)}{ϵ} \right) dt  + \sqrt{2 σ} dU(t), \quad &X_ϵ(0) = x_0, \\
  dY_ϵ(t) = -\frac{α}{ϵ} V'(X_ϵ(t)) - \frac{1}{ϵ^2} p'\left( Y_ϵ(t) \right) dt  + \sqrt{\frac{2 σ}{ϵ^2}} dU(t), \quad &Y_ϵ(0) = y_0.
\end{aligned}
```
Here, ``V`` is a large-scale potential and ``p`` a ``2π``-periodic function, see [`LDO`](@ref) or [`NLDO`](@ref). Note that ``Y_ϵ = X_ϵ/ϵ``. 
A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `x0::Real`:                               initial point ``x_0`` of slow process ``X_ϵ``.
- `y0::Real`:                               initial point ``y_0`` of fast process ``Y_ϵ``.
- `func_config::NTuple{4, Function}`:       collection of the functions ``V, V', p`` and ``p'``.
- `α::Real`:                                non-negative drift parameter ``α``.
- `σ::Real`:                                positive diffusion parameter ``σ``.
- `ϵ::Real=0.1`:                            positive small scale parameter ``ϵ``.
- `T::Real=100`:                            time horizon of time series.
- `dt::Real=1e-3`:                          time discretization step used in the Euler-Maruyama scheme.

---
# Examples
```julia
# quadratic potential V with sine oscillation p
T = 10.0
trajectory = Langevin(1.0, 3.0, func_config=LDO(), α=2.0, σ=1.0, ϵ=0.1, T=T)
fig = produce_trajectory(trajectory, T)
#save("trajectory_Langevin_process.pdf", fig)
```

---
See also [`LDO`](@ref), [`NLDO`](@ref).
"""
function Langevin(x0::Real, y0::Real; func_config::NTuple{4, Function}, α::Real, σ::Real, ϵ::Real=0.1, T::Real=100, dt::Real=1e-3)

  V_prime = func_config[2]
  p_prime = func_config[4]

  N = convert(Int64, T/dt) - 1

  X = Vector{Float64}(undef, N+1)
  Y = Vector{Float64}(undef, N+1)
  X[1] = x0
  Y[1] = y0
  
  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] + (-α*V_prime(X[k]) - 1/ϵ*p_prime(Y[k]))dt + sqrt(2σ)dW
    Y[k+1] = Y[k] + (-α/ϵ*V_prime(X[k]) - 1/ϵ^2*p_prime(Y[k]))dt + sqrt(2σ)/ϵ*dW
  end
  
  (X, Y)
end

# general limit process for overdamped Langevin process with large-scale potential V and fast oscillating part p

# corrective constant in effective limit equation where period L=2pi
@doc raw"""
    K(p::Function, σ::Real)

Return corrective constant of the cell problem of the homogenization in the overdamped Langevin case.

---
# Arguments
- `p`:                ``2π``-periodic function.
- `σ`:                positive diffusion parameter of slow process ``X_ϵ``.

---
See also [`Langevin`](@ref).
"""
function K(p::Function, σ::Real)
  Z1 = HCubature.hquadrature(x -> exp(p(x)/σ), 0, 2pi)[1]
  Z2 = HCubature.hquadrature(x -> exp(-p(x)/σ), 0, 2pi)[1]
  
  (2pi)^2/(Z1*Z2)
end

@doc raw"""
    Langevin(X0::Real; <keyword arguments>)

Return a one-dimensional overdamped limit Langevin process, homogenized from the multiscale overdamped Langevin process, starting at `X0` as a discretized time series.

The corresponding stochastic differential equation is defined for ``t \in [0,T]`` as
```math
\begin{aligned}
  dX(t) = -α K V'(X(t)) dt  + \sqrt{2 σ K} dW(t), \quad &X(0) = X_0.
\end{aligned}
```
Here, ``K`` is a corrective constant that comes from the cell problem of the homogenization, see also [`K`](@ref), and is computed inside the function.
A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `X0::Real`:                               initial point ``X_0`` of limit process ``X``.
- `func_config::NTuple{4, Function}`:       collection of the functions ``V, V', p`` and ``p'``.
- `α::Real`:                                non-negative drift parameter ``α``.
- `σ::Real`:                                positive diffusion parameter ``σ``.
- `T::Real=100`:                            time horizon of time series.
- `dt::Real=1e-3`:                          time discretization step used in the Euler-Maruyama scheme.

---
See also [`LDO`](@ref), [`NLDO`](@ref).
"""
function Langevin(X0::Real; func_config::NTuple{4, Function}, α::Real, σ::Real, T::Real=100, dt::Real=1e-3)
  
  V_prime = func_config[2]
  p = func_config[3]

  A = α*K(p, σ)
  Σ = σ*K(p, σ)

  N = convert(Int64, T/dt) - 1

  X = zeros(1, N+1)
  X[1] = X0
  
  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] - A*V_prime(X[k])dt + sqrt(2Σ)dW
  end
  
  X
end

# quadratic potential V with sine oscillation p
@doc raw"""
    LDO()

Return a tuple of functions used for the definition of [`Langevin`](@ref).

The returned functions are
```math
\begin{aligned}
  &V(x) = \frac12 x^2, \quad
  &V'(x) = x, \\
  &p(x) = \sin(x), \quad
  &p'(x) = \cos(x).
\end{aligned}
```
They yield a quadratic potential drift with a sine oscillation.

---
See also [`Langevin`](@ref).
"""
function LDO()
  V = x -> x^2/2
  V_prime = x -> x
  p = x -> sin(x)
  p_prime = x -> cos(x)

  (V, V_prime, p, p_prime)
end

# bistable potential V with sine oscillation p
@doc raw"""
    NLDO()

Return a tuple of functions used for the definition of [`Langevin`](@ref).

The returned functions are
```math
\begin{aligned}
  &V(x) = \frac14 x^4 - \frac12 x^2, \quad
  &V'(x) = x^3 - x, \\
  &p(x) = \sin(x), \quad
  &p'(x) = \cos(x).
\end{aligned}
```
This yields a bistable potential drift with a sine oscillation.

---
# Examples
```julia
# a slightly different potential (for illustrative reasons)
ϵ = 0.1
V(x) = -x^2 + x^4/12
p(x) = sin(x/ϵ)
x_range = range(-4,4,2000)

# create and adjust figure components; using CairoMakie.jl here
drift_fig = Figure(size=(3840,2160), fontsize = 50)
drift_ax = Axis(drift_fig[1, 1],
  # x-axis
  xlabel = L"x",
  xticks = LinearTicks(5),
  # y-axis
  yticks = LinearTicks(5),
)
Makie.xlims!(drift_ax, x_range[1], x_range[end])
colsize!(drift_fig.layout, 1, Aspect(1, 1.8))
  

V_line = lines!(drift_ax, x_range, map(V, x_range), linewidth = 10.0, color = (:darkgrey, 1.0), linestyle = :dash)
Vp_line = lines!(drift_ax, x_range, map(x->V(x)+p(x), x_range), linewidth = 3.0, color = (:black, 1.0))

axislegend(drift_ax,
[V_line, Vp_line],
[L"$x^4/12-x^2$", L"$x^4/12 - x^2 + \sin(x/%$ϵ)$"],
labelsize = 80
)

drift_fig
```

---
See also [`Langevin`](@ref).
"""
function NLDO()
  V = x -> -x^2/2 + x^4/4
  V_prime = x -> -x + x^3
  p = x -> sin(x)
  p_prime = x -> cos(x)

  (V, V_prime, p, p_prime)
end

## Overdamped Langevin process with quadratic potential and fast separable oscillating part in 2D ##

# general multiscale system for overdamped Langevin process with quadratic potential and fast separable oscillating part in 2D
@doc raw"""
    Langevin(x0::Vector{<:Real}, y0::Vector{<:Real}; <keyword arguments>)

Return a 4-dimensional overdamped Langevin process with a quadratic potential and a fast separable oscillating part starting at `(x0, y0)` as a discretized time series.

The corresponding stochastic differential equation is defined for ``t \in [0,T]`` as
```math
\begin{aligned}
  dX_ϵ(t) 
  &= \begin{pmatrix}
      dX^{(1)}_ϵ(t) \\[0.1cm]
      dX^{(2)}_ϵ(t)
    \end{pmatrix}
  = - M X_ϵ(t) - \frac{1}{ϵ} 
    \begin{pmatrix}
      p_1'\left(X^{(1)}_ϵ(t)/ϵ\right) \\[0.1cm]
      p_2'\left(X^{(2)}_ϵ(t)/ϵ\right)
    \end{pmatrix} dt  + \sqrt{2 σ} dU(t), \quad &X_ϵ(0) = x_0, \\
  dY_ϵ(t) &= \frac{X_ϵ}{ϵ}.
\end{aligned}
```
A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `x0::Vector{<:Real}`:                     initial point ``x_0 \in \mathbb{R}^{2}`` of slow process ``X_ϵ``.
- `y0::Vector{<:Real}`:                     initial point ``y_0 \in \mathbb{R}^{2}`` of slow process ``Y_ϵ``.
- `func_config::NTuple{2, Function}`:       collection of the ``2\pi``-periodic functions ``p_1'`` and ``p_2'``.
- `M::Array{<:Real, 2}`:                    positive definite drift matrix ``M \in \R^{2 \times 2}``.
- `σ::Real`:                                positive diffusion parameter ``σ``.
- `ϵ::Real=0.1`:                            positive small scale parameter ``ϵ``.
- `T::Real=100`:                            time horizon of time series.
- `dt::Real=1e-3`:                           time discretization step used in the Euler-Maruyama scheme.

---
# Examples
```julia
# quadratic potential V and fast separable oscillating part in 2D
trajectory = Langevin([-5.0, -5.0], [0.0, 0.0], func_config=(x-> cos(x), x -> 1/2*cos(x)), M=[4 2;2 3], σ=5.0, ϵ=0.05, T=10.0)
fig = produce_trajectory(trajectory)
#save("trajectory_Langevin_process_2D.pdf", fig)
```
"""
function Langevin(x0::Vector{<:Real}, y0::Vector{<:Real}; func_config::NTuple{2, Function}, M::Array{<:Real, 2}, σ::Real, ϵ::Real=0.1, T::Real=100, dt::Real=1e-3)
  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 2, N+1)
  Y = Array{Float64}(undef, 2, N+1)
  X[:,1] = x0
  Y[:,1] = y0
  
  for k in 1:N
    dW1 = sqrt(dt)*randn(1)[1]
    dW2 = sqrt(dt)*randn(1)[1]

    X[1,k+1] = X[1,k] - (M[1,1]X[1,k] + M[1,2]X[2,k] + 1/ϵ*func_config[1](Y[1,k]))dt + sqrt(2σ)dW1
    X[2,k+1] = X[2,k] - (M[2,1]X[1,k] + M[2,2]X[2,k] + 1/ϵ*func_config[2](Y[2,k]))dt + sqrt(2σ)dW2
    Y[1,k+1] = Y[1,k] - (M[1,1]/ϵ*X[1,k] + M[1,2]/ϵ*X[2,k] + 1/ϵ^2*func_config[1](Y[1,k]))dt + sqrt(2σ)/ϵ*dW1
    Y[2,k+1] = Y[2,k] - (M[2,1]/ϵ*X[1,k] + M[2,2]/ϵ*X[2,k] + 1/ϵ^2*func_config[2](Y[2,k]))dt + sqrt(2σ)/ϵ*dW2
  end
  
  (X, Y)
end

# general limit process for overdamped Langevin process with quadratic potential and fast separable oscillating part in 2D
@doc raw"""
    Langevin(X0::Vector{<:Real}; <keyword arguments>)

Return a 2-dimensional overdamped limit Langevin process, homogenized from the multiscale overdamped Langevin process,  starting at `X0` as a discretized time series.

The corresponding stochastic differential equation is defined for ``t \in [0,T]`` as
```math
\begin{aligned}
  dX(t) = - K M X(t) dt  + \sqrt{2 σ K} dW(t), \quad &X(0) = X_0.
\end{aligned}
```
Here, ``K \in \mathbb{R}^{2 \times 2}`` is a corrective constant that comes from the cell problem of the homogenization and is computed inside the function. 
A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `X0::Vector{<:Real}`:                     initial point ``X_0 \in \mathbb{R}^{2}`` of limit process ``X``.
- `func_config::NTuple{2, Function}`:       collection of the ``2\pi``-periodic functions ``p_1'`` and ``p_2'``.
- `M::Array{<:Real, 2}`:                    positive definite drift matrix ``M \in \mathbb{R}^{2 \times 2}``.
- `σ::Real`:                                positive diffusion parameter ``σ``.
- `T::Real=100`:                            time horizon of time series.
- `dt::Real=1e-3`:                           time discretization step used in the Euler-Maruyama scheme.
"""
function Langevin(X0::Vector{<:Real}; func_config::NTuple{2, Function}, M::Array{<:Real, 2},  σ::Real, T::Real=100, dt::Real=1e-3)

  CorrK = [K(func_config[1], σ) 0 ; 0 K(func_config[2], σ)]
  A = CorrK*M
  Σ = σ*CorrK

  N = convert(Int64, T/dt) - 1

  X = Array{Float64}(undef, 2, N+1)
  X[:,1] = X0
  
  for k in 1:N
    dW1 = sqrt(dt)*randn(1)[1]
    dW2 = sqrt(dt)*randn(1)[1]

    X[1,k+1] = X[1,k] - (A[1,1]X[1,k] + A[1,2]X[2,k])dt + sqrt(2Σ)[1,1]dW1 
    X[2,k+1] = X[2,k] - (A[2,1]X[1,k] + A[2,2]X[2,k])dt + sqrt(2Σ)[2,2]dW2 
  end
  
  X
end

## Truncated Burger's equation ##

# general multiscale system for truncated Burger's equation
@doc raw"""
    Burger(x0::Real, y0::Real, z0::Real; <keyword arguments>)

Return a three-dimensional process described through a truncated Burger's equation starting at `(x0, y0, z0)` as a discretized time series.

The corresponding stochastic differential equation is defined for ``t \in [0,T]`` as
```math
\begin{aligned}
  dX_ϵ(t) &= \left( ν X_ϵ(t) - \frac{1}{2ϵ} (X_ϵ(t)Y_ϵ(t) + Y_ϵ(t)Z_ϵ(t)) \right) dt,                                                 \quad &X_ϵ(0) = x_0,  \\
  dY_ϵ(t) &= \left( ν Y_ϵ(t) - \frac{3}{ϵ^2} Y_ϵ(t) - \frac{1}{2ϵ} (2 X_ϵ(t)Z_ϵ(t) - X_ϵ(t)^2) \right) dt + \frac{q_1}{ϵ} dV_1(t),    \quad &Y_ϵ(0) = y_0,  \\
  dZ_ϵ(t) &= \left( ν Z_ϵ(t) - \frac{8}{ϵ^2} Y_ϵ(t) - \frac{3}{2ϵ} X_ϵ(t)Y_ϵ(t) \right) dt + \frac{q_2}{ϵ} dV_2(t),                   \quad &Z_ϵ(0) = z_0.
\end{aligned}
```
A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `x0::Real`:         initial point ``x_0`` of slow process ``X_ϵ``.
- `y0::Real`:         initial point ``y_0`` of fast process ``Y_ϵ``.
- `z0::Real`:         initial point ``z_0`` of fast process ``Z_ϵ``.
- `ν::Real`:          positive parameter ``ν``.
- `q1::Real`:         positive parameter ``q_1``.
- `q2::Real`:         positive parameter ``q_2``.
- `ϵ::Real=0.1`:      positive small scale parameter ``ϵ``.
- `T::Real=100`:      time horizon of time series.
- `dt::Real=1e-3`:    time discretization step used in the Euler-Maruyama scheme.
"""
function Burger(x0::Real, y0::Real, z0::Real; ν::Real, q1::Real, q2::Real, ϵ::Real=0.1, T::Real=100, dt::Real=1e-3)
  
  N = convert(Int64, T/dt) - 1

  X = Vector{Float64}(undef, N+1)
  Y = Vector{Float64}(undef, N+1)
  Z = Vector{Float64}(undef, N+1)
  X[1] = x0
  Y[1] = y0
  Z[1] = z0
  
  for k in 1:N
    dW1 = sqrt(dt)*randn(1)[1]
    dW2 = sqrt(dt)*randn(1)[1]

    X[k+1] = X[k] + (ν*X[k] - 1/2ϵ*(X[k]Y[k] + Y[k]Z[k]))dt
    Y[k+1] = Y[k] + (ν*Y[k] - 3/ϵ^2*Y[k] - 1/2ϵ*(2X[k]Z[k] - X[k]^2))dt + q1/ϵ*dW1
    Z[k+1] = Z[k] + (ν*Z[k] - 8/ϵ^2*Z[k]  + 3/2ϵ*X[k]Y[k])dt + q2/ϵ*dW2
  end
  
  (X, Y, Z)
end

# general limit process for truncated Burger's equation
@doc raw"""
    Burger(X0::Real; <keyword arguments>)

Return a one-dimensional process described through a limit truncated Burger's equation, homogenized from the multiscale truncated Burger's equation, starting at `X0` as a discretized time series.

The corresponding stochastic differential equation is defined for ``t \in [0,T]`` as
```math
\begin{aligned}
  dX(t) = \left(AX(t) - BX(t)^3\right) dt  + \sqrt{σ_a + σ_bX(t)^2} dW(t), \quad X(0) = X_0,
\end{aligned}
```
with the paraters
```math
\begin{aligned}
  A = ν + \frac{q_1^2}{396} + \frac{q_2^2}{352}, \quad B = \frac{1}{12}, \quad σ_a = \frac{q_1^2 q_2^2}{2112}, \quad σ_b = \frac{q_1^2}{36}.
\end{aligned}
```
A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `X0::Real`:         initial point ``X_0`` of limit process ``X``.
- `ν::Real`:          positive parameter ``ν``.
- `q1::Real`:         positive parameter ``q_1``.
- `q2::Real`:         positive parameter ``q_2``.
- `T::Real=100`:      time horizon of time series.
- `dt::Real=1e-3`:     time discretization step used in the Euler-Maruyama scheme.
"""
function Burger(X0::Real; ν::Real, q1, q2, T=100, dt=1e-3)

  A = ν + q1^2/396 + q2^2/352
  B = 1/12
  σ_a = q1^2*q2^2/2112
  σ_b = q1^2/36

  N = convert(Int64, T/dt) - 1

  X = Vector{Float64}(undef, N+1)
  X[1] = X0

  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] + (A*X[k] - B*X[k]^3)dt + sqrt(σ_a + σ_b*X[k]^2)dW
  end

  X
end

## Fast chaotic noise ##

# general multiscale system for fast chaotic noise solved with DifferentialEquations.jl using a fourth order Runge-Kutta scheme
@doc raw"""
    Fast_chaotic(xy0; <keyword arguments>)

Return the 4-dimensional solution path of a fast chaotic noise system starting at `xy0` as a discretized time series.

The corresponding ordinary differential equation is defined for ``t \in [0,T]`` as
```math
\begin{aligned}
  \frac{dX_\epsilon}{dt} &= AX - BX^3 + \frac{\lambda}{\epsilon} Y_\epsilon^{(2)}, \quad &X_ϵ(0) = x_0, \\
  \frac{dY_\epsilon^{(1)}}{dt} &= \frac{10}{\epsilon^2} \left( Y_\epsilon^{(2)} - Y_\epsilon^{(1)} \right), \quad &Y_\epsilon^{(1)}(0) = y_0^{(1)}, \\
  \frac{dY_\epsilon^{(2)}}{dt} &= \frac{1}{\epsilon^2} \left( 28 Y_\epsilon^{(1)} - Y_\epsilon^{(2)} - Y_\epsilon^{(1)}Y_\epsilon^{(3)} \right), \quad &Y_\epsilon^{(2)}(0) = y_0^{(2)}, \\
  \frac{dY_\epsilon^{(3)}}{dt} &= \frac{1}{\epsilon^2} \left( Y_\epsilon^{(1)}Y_\epsilon^{(2)} - \frac83 Y_\epsilon^{(3)} \right), \quad &Y_\epsilon^{(3)}(0) = y_0^{(3)}.
\end{aligned}
```
The ODE is solved with a fourth order Runge-Kutta scheme of the [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)  package.

---
# Arguments
- `xy0::Vector{<:Real}`:  initial point ``(x_0, y_0^{(1)}, y_0^{(2)}, y_0^{(3)}) \in \mathbb{R}^{4}``.
- `A::Real`:              positive parameter ``A``.
- `B::Real`:              positive parameter ``B``.
- `λ::Real`:              positive parameter ``λ``.
- `T::Real=100`:          time horizon of time series.
- `dt::Real=1e-3`:        time discretization step used in the ODE solver.
"""
function Fast_chaotic(xy0::Vector{<:Real}; A::Real, B::Real, λ::Real, ϵ::Real=0.1, T::Real=100, dt::Real=1e-3)

  # defining ode; specifically written for DifferentialEquations.jl
  function Fast_chaotic_ode!(du, u, p, t)
    du[1] = p[1]*u[1] - p[2]*u[1]^3 + p[3]/p[4]*u[3]
    du[2] = 10/p[4]^2*(u[3] - u[2])
    du[3] = 1/p[4]^2*(28u[2] - u[3] - u[2]*u[4])
    du[4] = 1/p[4]^2*(u[2]*u[3] - 8/3*u[4])
  end

  p = [A, B, λ, ϵ]
  tspan = (0.0, T)

  prob = ODEProblem(Fast_chaotic_ode!, xy0, tspan, p)
  sol = solve(prob, Tsit5(), saveat = dt, maxiters = 1e8)

  X = sol[1,:]
  pop!(X)
  X
end

# general limit process for fast chaotic noise
@doc raw"""
    Fast_chaotic(X0::Real; <keyword arguments>)

Return a one-dimensional process described through a limit fast chaotic noise, homogenized from the multiscale fast chaotic noise system, starting at `X0` as a discretized time series.

The corresponding stochastic differential equation is defined for ``t \in [0,T]`` as
```math
\begin{aligned}
  dX(t) = \left(AX(t) - BX(t)^3\right) dt  + \sqrt{σ} dW(t), \quad X(0) = X_0,
\end{aligned}
```
The diffusion parameter ``σ`` is given by the Green-Kubo formula 
```math
\begin{aligned}
  \sigma = \frac{\lambda^2}{2} \int_0^\infty \lim_{T \rightarrow \infty} \frac1T \int_0^T Y_{ϵ=1}^{(2)}(s) Y_{ϵ=1}^{(2)}(s+t) \, ds \, dt,
\end{aligned}
```
which requires to be numerically computed or estimated through data. A simple Euler-Maruyama discretization is implemented for the generation of the time series.

---
# Arguments
- `X0::Real`:         initial point ``X_0`` of limit process ``X``.
- `A::Real`:          positive drift parameter ``A``.
- `B::Real`:          positive drift parameter ``B``.
- `σ::Real`:          positive diffusion parameter ``σ``.
- `T::Real=100`:      time horizon of time series.
- `dt::Real=1e-3`:    time discretization step used in the Euler-Maruyama scheme.
"""
function Fast_chaotic(X0::Real; A::Real, B::Real, σ::Real, T::Real=100, dt::Real=1e-3)

  N = convert(Int64, T/dt) - 1

  X = Vector{Float64}(undef, N+1)
  X[1] = X0

  for k in 1:N
    dW = sqrt(dt)*randn(1)[1]
    X[k+1] = X[k] + (A*X[k] - B*X[k]^3)dt + sqrt(σ)dW
  end

  X
end

## functions for generating a trajectory plot ##
@doc raw"""
    produce_trajectory(trajectory, T)

Generate a plot of a 2-dimensional (slow + fast scale process) multiscale time series of length `T` using the [CairoMakie.jl](https://github.com/MakieOrg/Makie.jl/tree/master/CairoMakie) package.

---
# Arguments
- `trajectory::NTuple{2, Vector{<:Real}}`:   2-dimensional time series of length `T`
- `T::Real`:                                 time horizon of time series.

---
# Examples
```julia
# quadratic potential V with sine oscillation p
T = 10.0
trajectory = Langevin(1.0, -4.0, func_config=LDO(), α=2.0, σ=1.0, ϵ=0.1, T=T)
fig = produce_trajectory(trajectory, T)
```
"""
function produce_trajectory(trajectory::NTuple{2, Vector{<:Real}}, T::Real)

  N = length(trajectory[1])
  T_range = range(0, T, N)

  # create and adjust figure components
  process_fig = Figure(size=(3840,2160), fontsize = 50)
  process_ax = Axis(process_fig[1, 1],
      # x-axis
      xlabel = L"T",
      xticks = LinearTicks(5),
      # y-axis
      yticks = LinearTicks(10),
  )
  Makie.xlims!(process_ax, 0.0, T)
  colsize!(process_fig.layout, 1, Aspect(1, 1.8))
  

  X_ϵ_line = lines!(process_ax, T_range, trajectory[1][:], linewidth = 3.0, color = (:purple, 1.0))
  Y_ϵ_line = lines!(process_ax, T_range, trajectory[2][:], linewidth = 3.0, color = (:grey, 0.4))
  
  axislegend(process_ax,
      [X_ϵ_line, Y_ϵ_line],
      [L"Slow process $X_ϵ$", L"Fast process $Y_ϵ$"],
      labelsize = 80
  )

  process_fig
end

@doc raw"""
    produce_trajectory(trajectory)

Generate a plot of a 4-dimensional (slow + fast scale process) multiscale time series using the [CairoMakie.jl](https://github.com/MakieOrg/Makie.jl/tree/master/CairoMakie) package.

---
# Arguments
- `trajectory::NTuple{2, Array{<:Real, 2}}`:   4-dimensional time series

---
# Examples
```julia
# quadratic potential V and fast separable oscillating part in 2D
trajectory = Langevin([-5.0, -5.0], [10.0, 10.0], func_config=(x-> cos(x), x -> 1/2*cos(x)), M=[4 2;2 3], σ=5.0, ϵ=0.05, T=10.0)
fig = produce_trajectory(trajectory)
```
"""
function produce_trajectory(trajectory::NTuple{2, Array{<:Real, 2}})

  # create and adjust figure components
  process_fig = Figure(size=(3840,2160), fontsize = 50)
  process_ax = Axis(process_fig[1, 1],
      # x-axis
      xticks = LinearTicks(10),
      # y-axis
      yticks = LinearTicks(10),
  )
  colsize!(process_fig.layout, 1, Aspect(1, 1.8))
  

  X_ϵ_line = lines!(process_ax, trajectory[1][1,:], trajectory[1][2,:], linewidth = 3.0, color = (:purple, 1.0))
  Y_ϵ_line = lines!(process_ax, trajectory[2][1,:], trajectory[2][2,:], linewidth = 3.0, color = (:grey, 0.4))
  
  axislegend(process_ax,
      [X_ϵ_line, Y_ϵ_line],
      [L"Slow process $X_ϵ$", L"Fast process $Y_ϵ$"],
      labelsize = 80
  )

  process_fig
end

function produce_trajectory(trajectory::Vector{<:Real}, T::Real)

  N = length(trajectory)
  T_range = range(0, T, N)

  # create and adjust figure components
  process_fig = Figure(size=(3840,2160), fontsize = 50)
  process_ax = Axis(process_fig[1, 1],
      # x-axis
      xlabel = L"T",
      xticks = LinearTicks(5),
      # y-axis
      yticks = LinearTicks(10),
  )
  Makie.xlims!(process_ax, 0.0, T)
  colsize!(process_fig.layout, 1, Aspect(1, 1.8))
  
  lines!(process_ax, T_range, trajectory, linewidth = 3.0, color = (:pink, 1.0))

  process_fig
end