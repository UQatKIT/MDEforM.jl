########################################################################################################
########################################################################################################
# The main object of computation. The optimization task of the MDE is performed here for various
# examples of defining limit models.
########################################################################################################
########################################################################################################
# Jaroslav Borodavka, 21.08.2024

########################################################################################################
## optimizers for the MDE
########################################################################################################

## limit model: 1-dimensional Ornstein-Uhlenbeck process / Langevin process with a quadratic potential ##

@doc raw"""
    MDE(data::Vector{<:Real}, limit_model::String, prior_parameter::Real, ϑ_initial::Real; verbose::Bool=false)

Return MDE value for given `data` in form of a time series, a defining `limit_model`, a prior estimation parameter `prior_parameter`, and an initial point `ϑ_initial`
of the involved optimization procedure.

The optimization task
```math
\begin{aligned}
  \argmin_{\vartheta \in \Theta} \Delta_T(X_\epsilon, \vartheta, \Sigma, V),
\end{aligned}
```
is implemented and solved with the Julia package [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/). Here, ``X_ϵ`` is a one-dimensional time series of length ``T``,
obtained from a multiscale SDE, ``\Delta_T`` is the associated cost functional of the MDE, see [`Δ`](@ref), ``\Sigma`` is the `prior_parameter`, and ``V`` is a potential that
is, in the given case, quadratic, i.e. ``V(x)=x^2/2``.

---
# Arguments
- `data::Vector{<:Real}`:       one-dimensional time series ``X_ϵ``.
- `limit_model::String`:        defining limit model; thus far only supports "Langevin" and "Fast Chaotic Noise".
- `prior_parameter::Real`:      prior estimation parameter; limit diffusion parameter in the "Langevin" case and limit drift parameter in the "Fast Chaotic Noise" case.
- `ϑ_initial::Real`:            initial point of the numerical optimization procedure.
- 'verbose::Bool=false':        if `verbose = true`, then detailed information on the optimization will be printed in real-time.

---
# Examples
```julia-repl
julia> using MDEforM
julia> limit_drift_parameter = 1.0
julia> data = Fast_chaotic([1.0, 1.0, 1.0, 1.0], A=-limit_drift_parameter, B=0.0, λ=2/45, ϵ=0.1, T=1000)
julia> MDE(data, "Fast Chaotic Noise", limit_drift_parameter, 10.0)
```

---
See also [`Δ_grad_ϑ`](@ref).
"""
function MDE(data::Vector{<:Real}, limit_model::String, prior_parameter::Real, ϑ_initial::Real; verbose::Bool=false)

    # specifying boundary constraints, since the parameter lies in (0, ∞)
    lower = 0.0
    upper = Inf
    # optimizer for box-constrained optimization
    inner_optimizer = LBFGS()

    println("⎔ GD initial point: ϑ₀ = $(round(ϑ_initial, digits = 3))")

    # cost functional, gradient and optimization depending on the limit model
    if limit_model == "Langevin"
        J_L(ϑ) = Δ(data, ϑ, prior_parameter)
        ∇J_L(ϑ) = Δ_grad_ϑ(data, ϑ, prior_parameter)

        # gradient of cost functional, specifically written for Optim.optimize
        function gradient!(storage, ϑ)
            storage[1] = ∇J_L(ϑ[1])
        end

        # optimize with Optim.jl
        optim_res = optimize(ϑ -> J_L(first(ϑ)), gradient!, [lower], [upper], [ϑ_initial], 
        Fminbox(inner_optimizer), Optim.Options(show_trace = verbose, g_tol=1e-6))
    elseif limit_model == "Fast Chaotic Noise"
        J_FCN(ϑ) = Δ(data, prior_parameter, ϑ/2)
        
        # optimize with Optim.jl; using forward automatic differentiation for gradient
        optim_res = optimize(ϑ -> J_FCN(first(ϑ)), [lower], [upper], [ϑ_initial], 
        Fminbox(inner_optimizer), Optim.Options(show_trace = verbose, g_tol=1e-6), autodiff = AutoForwardDiff())
    end
 
    @show optim_res
    println("✪ MDE value: $(round(Optim.minimizer(optim_res)[1], digits = 6))")
    Optim.minimizer(optim_res)[1]
end

## limit model: 1-dimensional Langevin process with a general potential V ##

@doc raw"""
    MDE(data::Vector{<:Real}, limit_model::String, V::Function, prior_parameter::Real, ϑ_initial::Real; verbose::Bool=false)

Return MDE value for given `data` in form of a time series, a defining `limit_model`, a general potential `V`, a prior estimation parameter `prior_parameter`, and an initial point `ϑ_initial`
of the involved optimization procedure.

The optimization task
```math
\begin{aligned}
  \argmin_{\vartheta \in \Theta} \Delta_T(X_\epsilon, \vartheta, \Sigma, V),
\end{aligned}
```
is implemented and solved with the Julia package [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/). Here, ``X_ϵ`` is a one-dimensional time series of length ``T``,
obtained from a multiscale SDE, ``\Delta_T`` is the associated cost functional of the MDE, see [`Δ`](@ref), ``\Sigma`` is the `prior_parameter`, and ``V`` is a general potential.

---
# Arguments
- `data::Vector{<:Real}`:       one-dimensional time series ``X_ϵ``.
- `limit_model::String`:        defining limit model; thus far only supports "Langevin" and "Fast Chaotic Noise".
- `V::Function`:                potential ``V`` that defines the invariant density of the limit model.
- `prior_parameter::Real`:      prior estimation parameter; limit diffusion parameter in the "Langevin" case and limit drift parameter in the "Fast Chaotic Noise" case.
- `ϑ_initial::Real`:            initial point of the numerical optimization procedure.
- 'verbose::Bool=false':        if `verbose = true`, then detailed information on the optimization will be printed in real-time.

---
# Examples
```julia-repl
julia> using MDEforM
julia> data_eps = Langevin(10.0, 5.0, func_config=NLDO(), α=2.0, σ=1.0, ϵ=0.1, T=2500)[1]
julia> data_limit = Langevin(10.0, func_config=NLDO(), α=2.0, σ=1.0, T=2500)
julia> V = NLDO()[1]
julia> limit_diffusion_parameter = 1.0*K(NLDO()[3], 1.0)
julia> MDE(data_limit[:], "Langevin", V, limit_diffusion_parameter, 10.0)
julia> MDE(data_eps, "Langevin", V, limit_diffusion_parameter, 10.0)            # true parameter = 2.0*K(NLDO()[3], 1.0) ≈ 1.248
```

```julia-repl
julia> using MDEforM
julia> limit_drift_parameter = 1.0
julia> data = Fast_chaotic([1.0, 1.0, 1.0, 1.0], A=limit_drift_parameter, B=1.0, λ=2/45, ϵ=10^(-3/2), T=100)
julia> V = NLDO()[1]
julia> MDE(data, "Fast Chaotic Noise", V, limit_drift_parameter, 0.8)
```

---
See also [`Δ_grad_ϑ`](@ref), [`Δ_grad_Σ`](@ref), [`μ`](@ref).
"""
function MDE(data::Vector{<:Real}, limit_model::String, V::Function, prior_parameter::Real, ϑ_initial::Real; verbose::Bool=false)
    # specifying boundary constraints, since the parameter lies in (0, ∞)
    lower = 0.0
    upper = Inf
    # optimizer for box-constrained optimization
    inner_optimizer = LBFGS()

    println("⎔ GD initial point: ϑ₀ = $(round(ϑ_initial, digits = 3))")

    # cost functional, gradient and optimization depending on the limit model
    if limit_model == "Langevin"
        J_L(ϑ) = Δ(data, ϑ, prior_parameter, V)
        ∇J_L(ϑ) = Δ_grad_ϑ(data, ϑ, prior_parameter, V)

        # gradient of cost functional, specifically written for Optim.optimize
        function gradient_L!(storage, ϑ)
            storage[1] = ∇J_L(ϑ[1])
        end

        # optimize with Optim.jl
        optim_res = optimize(ϑ -> J_L(first(ϑ)), gradient_L!, [lower], [upper], [ϑ_initial], 
        Fminbox(inner_optimizer), Optim.Options(show_trace = verbose, g_tol=1e-6))
    elseif limit_model == "Fast Chaotic Noise"
        J_FCN(ϑ) = Δ(data, prior_parameter, ϑ/2, V)
        ∇J_FCN(ϑ) = Δ_grad_Σ(data, prior_parameter, ϑ/2, V)

        # gradient of cost functional, specifically written for Optim.optimize
        function gradient_FCN!(storage, ϑ)
            storage[1] = ∇J_FCN(ϑ[1])
        end
        
        # optimize with Optim.jl
        optim_res = optimize(ϑ -> J_FCN(first(ϑ)), gradient_FCN!, [lower], [upper], [ϑ_initial], 
        Fminbox(inner_optimizer), Optim.Options(show_trace = verbose, g_tol=1e-6))
    end
 
    @show optim_res
    println("✪ MDE value: $(round(Optim.minimizer(optim_res)[1], digits = 6))")
    Optim.minimizer(optim_res)[1]
end

## limit model: 2-dimensional Ornstein-Uhlenbeck process / Langevin process with a quadratic potential ##

@doc raw"""
    MDE(data::Array{<:Real, 2}, limit_diffusion::Array{<:Real, 2}, ϑ_initial::Array{<:Real, 2}; verbose::Bool=false)

Return MDE value for given `data` in form of a time series, a prior estimation parameter `limit_diffusion`, and an initial point `ϑ_initial`
of the involved optimization procedure.

The optimization task
```math
\begin{aligned}
  \argmin_{\vartheta \in \Theta} \Delta_T(X_\epsilon, \vartheta, \Sigma, V),
\end{aligned}
```
is implemented and solved with the Julia package [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/). Here, ``X_ϵ`` is a two-dimensional time series of length ``T``,
obtained from a multiscale SDE, ``\Delta_T`` is the associated cost functional of the MDE, see [`Δ`](@ref), ``\Sigma`` is the prior 2x2 limit diffusion matrix.
The initial point `ϑ_initial` must satisfy certain parameter constraints since the optimization has nonlinear contraints, see manuscript or source code.

---
# Arguments
- `data::Array{<:Real, 2}`:                 two-dimensional time series ``X_ϵ``.
- `limit_diffusion::Array{<:Real, 2}`:      positive definite limit diffusion matrix ``\Sigma \in \mathbb{R}^{2 \times 2}``.
- `ϑ_initial::Array{<:Real, 2}`:            initial point ``\vartheta_0 \in \mathbb{R}^{2 \times 2}`` of the numerical optimization procedure.
- 'verbose::Bool=false':                    if `verbose = true`, then detailed information on the optimization will be printed in real-time.

---
# Examples
```julia-repl
julia> using MDEforM
julia> M = [4 2;2 3]
julia> σ = 1.5               
julia> p1, p2 = (x -> sin(x), x -> 1/2*sin(x))     
julia> p1_prime, p2_prime = (x-> cos(x), x -> 1/2*cos(x))
julia> CorrK = [K(p1, σ) 0 ; 0 K(p2, σ)]
julia> A = CorrK*M # true parameter; the parameter that ought to be estimated
julia> Σ = σ*CorrK
julia> data = Langevin([-5.0, -5.0], [0.0, 0.0], func_config=(p1_prime, p2_prime), M=M, σ=σ, ϵ=0.1, T=1000)[1]
julia> ϑ_initial = [3.0 0.5*Σ[1]; 0.5*Σ[4] 6.0]
julia> MDE(data, Σ, ϑ_initial)
```

---
See also [`Δ`](@ref).
"""
function MDE(data::Array{<:Real, 2}, limit_diffusion::Array{<:Real, 2}, ϑ_initial::Array{<:Real, 2}; verbose::Bool=false)

    # cost functional and gradient
    J(ϑ) = Δ(data, ϑ, limit_diffusion)

    # diagonal elements of the limit diffusion matrix
    Σ11 = limit_diffusion[1,1]
    Σ22 = limit_diffusion[2,2]

    # nonlinear constraints on the parameter space
    function con_func!(c, A)
        c[1] = A[1]                         # First constraint
        c[2] = A[1]A[4] - A[3]A[2]          # Second constraint
        c[3] = A[2]/Σ11 - A[3]/Σ22          # Third constraint
        c
    end

    function con_jacobian!(Jac, A)
        # First constraint
        Jac[1,1] = 1.0
        Jac[1,2] = 0.0
        Jac[1,3] = 0.0
        Jac[1,4] = 0.0
        # Second constraint 
        Jac[2,1] = A[4]
        Jac[2,2] = -A[3]
        Jac[2,3] = -A[2]
        Jac[2,4] = A[1]
        # Third constraint 
        Jac[3,1] = 0.0
        Jac[3,2] = 1/Σ11
        Jac[3,3] = -1/Σ22
        Jac[3,4] = 0.0
        Jac
    end

    function con_hessian!(Hess, A, λ)
        # only the second nonlinear constraint has nontrivial contribution to the hessian
        Hess[1,4] += λ[2]
        Hess[2,3] += -λ[2]
        Hess[3,2] += -λ[2]
        Hess[4,1] += λ[2]
        Hess
    end

    # constraint check for initial point
    if !(con_func!(fill(0.0, 3), ϑ_initial) > fill(0.0, 3))
        return @warn "The initial point for the optimization $(ϑ_initial) does not fulfill the parameter constraints. Type ?MDE for info."
    end

    # parameter bounds, constraint bounds and problem definition for Optim.jl
    lx = fill(-Inf, 4); ux = fill(Inf, 4)
    lc = [0.0, 0.0, 0.0]; uc = [Inf, Inf, 0.0]
    dfc = TwiceDifferentiableConstraints(con_func!, con_jacobian!, con_hessian!, lx, ux, lc, uc)

    # optimize with Optim.jl, using forward autodiff (forward automatic differentiation)
    println("⎔ GD initial point: ϑ₀ = $(round.(ϑ_initial, digits = 3))")
    optim_res = optimize(J, dfc, vec(ϑ_initial), IPNewton(),
    Optim.Options(allow_f_increases=true, show_trace=verbose, g_tol=1e-6), autodiff=:AutoForwardDiff())
    @show optim_res

    # result as a 2x2 matrix instead of a vector; transpose result of reshape() to get correct matrix
    result_matrix = reshape(Optim.minimizer(optim_res), 2, 2)'
    println("✪ MDE value: $(round.(result_matrix, digits = 6))")
    result_matrix
end

