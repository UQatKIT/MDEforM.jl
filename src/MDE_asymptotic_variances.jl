########################################################################################################
########################################################################################################
# This script computes the asymptotic variance of the MDE in the Langevin case with quadratic and
# quartic potential, see numerics section of the main manuscript.
########################################################################################################
########################################################################################################
# Jaroslav Borodavka, 20.08.2024

########################################################################################################
## asymptotic variances for the MDE ##
########################################################################################################

## quadratic potential QdP ##

# invariant density with quadratic potential
μ_QdP(x, ϑ, Σ) = μ(x, ϑ, Σ, x -> x^2/2)

# RHS of limit Poisson equation -AΦ=h
function h_QdP(x, ϑ, Σ)
  β = k(0)[2]
  σ_1 = sqrt(β^2*ϑ/(ϑ + Σ*β^2))
  σ_2 = sqrt(β^2*ϑ/(ϑ + 2Σ*β^2))
  Σ/(2ϑ^2*β)*((σ_1^3*(1 - σ_1^2*x^2)*exp(-σ_1^2*x^2/2)) - σ_2^3)
end

# derivative of solution of limit Poisson equation -AΦ=h
function Φ_prime_QdP(x, ϑ, Σ)
  f(y) = h_QdP(y, ϑ, Σ)*μ_QdP(y, ϑ, Σ)
  res = 0.0
    # case distinction where we switch the integration domain, cf. proof of Lemma B.13, to make numerical 
    # computation feasible since we divide a very small number by a large number 
    if x > 0
      return res = -1/(Σ*μ_QdP(x, ϑ, Σ))*QuadGK.quadgk(f, x, Inf)[1]  # QuadGK is faster than HCubature here
    elseif x == 0 # evaluation at x=0 takes forever, but the result, in fact, equals zero
        return res = 0.0
    else
      return res = 1/(Σ*μ_QdP(x, ϑ, Σ))*QuadGK.quadgk(f, -Inf, x)[1]
    end
end

# asymptotic variance factor, Dirichlet form corresponding to generator A
function τ²_QdP(ϑ, Σ)
  f(x) = 2Σ*Φ_prime_QdP(x, ϑ, Σ)^2*μ_QdP(x, ϑ, Σ)
  # functions are symmetric in the considered cases; upper integration limit must be chosen ad-hoc;
  # notice that the integrand decays rapidly to zero
  2hquadrature(f, 0, 20)[1]  
end

# Fisher information term
function J_fisher_QdP(ϑ, Σ)
  β = k(0)[2]
  σ_2 = sqrt(β^2*ϑ/(ϑ + 2Σ*β^2))
  3/(4β)*(Σ/ϑ^2)^2*σ_2^5
end

# asymptotic variance
@doc raw"""
    Σ_∞_QdP(ϑ, Σ)

Return the asymptotic variance of the MDE in the multiscale overdamped Langevin drift parameter estimation problem with a quadratic potential, drift parameter `ϑ`, and diffusion parameter `Σ`.

The asymptotic variance is given by
```math
\begin{aligned}
  \frac{\tau^2(\vartheta, \Sigma)}{J(\vartheta, \Sigma)^2} = \frac{2\Sigma}{J(\vartheta, \Sigma)^{2}} \int_\R |\Phi'(x)|^2 \mu(x, \vartheta, \Sigma, V) \, dx,
\end{aligned}
```
where
```math
\begin{aligned}
  J(\vartheta, \Sigma) = \| \partial_\vartheta \mathscr{C}_{\vartheta} \|_{L^2(\varphi)}, \quad \Phi'(x) = \frac{1}{\Sigma \mu(x, \vartheta, \Sigma, V)} \int_{-\infty}^x h(z) \mu(x, \vartheta, \Sigma, V) \, dz, \quad x \in \R.
\end{aligned}
```
Here ``\mu(x, \vartheta, \Sigma, V)`` corresponds to [`μ`](@ref) with ``V(x) = x^2/2``. In the case of such a quadratic potential ``V`` it holds
```math
\begin{aligned}
  J(\vartheta, \Sigma) = \frac{3}{4 \beta} \left( \frac{\Sigma}{\vartheta^2} \right)^2 \sigma_2^5, \quad h(z) = \frac{\Sigma}{2 \vartheta^2 \beta} \left[ \sigma_1^3 (1 - \sigma_1^2 z^2) \exp \left( -\frac{\sigma_1^2 z^2}{2} \right) - \sigma_2^3 \right], \quad z \in \R,
\end{aligned}
```
with
```math
\begin{aligned}
      \sigma_1^2 := \frac{\beta^2 \vartheta}{\vartheta + \Sigma\beta^2 }, \quad \sigma_2^2 := \frac{\beta^2 \vartheta}{\vartheta + 2 \Sigma\beta^2 }.
\end{aligned}
```

!!! warning 
    The running times for a single evaluation can take some time due to the involved integrations, but it usually needs
    to be computed only once.

---
# Arguments
- `ϑ::Real`:                    positive drift coefficient ``\vartheta``.
- `Σ::Real`:                    positive diffusion coefficient ``\Sigma``.

---
# Examples
```julia-repl
julia> Σ_∞_QdP(1.2, 0.6)
```

---
See also [`μ`](@ref), [`Σ_∞_QrP`](@ref).
"""
function Σ_∞_QdP(ϑ, Σ)
  τ²_QdP(ϑ, Σ)/J_fisher_QdP(ϑ, Σ)^2
end

## quartic potential QrP ##

# invariant density with quartic potential
V(x) = x^4/4-x^2/2
μ_QrP(x, ϑ, Σ) = μ(x, ϑ, Σ, V)

# convolution term in Fisher information and h
inner_convolution(x, ϑ, Σ) = hquadrature(y -> ∂ϑ_μ(t(y), ϑ, Σ, V)k(x-t(y))[1]dt(y), -1, 1)[1]
  
# asymptotic variance factor, Dirichlet form corresponding to generator A
function τ²_QrP(ϑ, Σ)
  # constant term of RHS of limit Poisson equation -AΦ=h
  function h_QrP_const(ϑ, Σ)
    f(y) = inner_convolution(t(y), ϑ, Σ)μ_QrP(t(y), ϑ, Σ)dt(y)
    # function is symmetric
    2hquadrature(f, 0, 1)[1]
  end

  # define constant factor to reduce running time; main reason why all these functions are inside τ²_QrP
  h_QrP_const_val = h_QrP_const(ϑ, Σ)

  # RHS of limit Poisson equation -AΦ=h
  h_QrP(x, ϑ, Σ) = inner_convolution(x, ϑ, Σ) - h_QrP_const_val

  # derivative of solution of limit Poisson equation -AΦ=h
  function Φ_prime_QrP(x, ϑ, Σ)
    f(y) = h_QrP(y, ϑ, Σ)*μ_QrP(y, ϑ, Σ)
    res = 0.0
    # case distinction where we switch the integration domain, cf. proof of Lemma B.13, to make numerical 
    # computation feasible since we divide a very small number by a large number 
    if x > 0
      return res = -1/(Σ*μ_QrP(x, ϑ, Σ))*QuadGK.quadgk(f, x, Inf)[1]  # QuadGK is faster than HCubature here
    elseif x == 0 # evaluation at x=0 takes forever, but the result, in fact, equals zero
      return res = 0.0
    else
      return res = 1/(Σ*μ_QrP(x, ϑ, Σ))*QuadGK.quadgk(f, -Inf, x)[1]
    end
  end

  f(x) = 2Σ*Φ_prime_QrP(x, ϑ, Σ)^2*μ_QrP(x, ϑ, Σ)
  # functions are symmetric in the considered cases; upper integration limit must be chosen ad-hoc;
  # notice that the integrand decays rapidly to zero
  2hquadrature(f, 0, 2)[1]  
end
  
# Fisher information term
function J_fisher_QrP(ϑ, Σ)
    f(x) = inner_convolution(t(x), ϑ, Σ)∂ϑ_μ(t(x), ϑ, Σ, V)dt(x)
    # functions are symmetric
    2hquadrature(f, 0, 1)[1]
end
  
# asymptotic variance
@doc raw"""
    Σ_∞_QrP(ϑ, Σ)

Return the asymptotic variance of the MDE in the multiscale overdamped Langevin drift parameter estimation problem with a quartic potential, drift parameter `ϑ`, and diffusion parameter `Σ`.

The asymptotic variance is given by
```math
\begin{aligned}
  \frac{\tau^2(\vartheta, \Sigma)}{J(\vartheta, \Sigma)^2} = \frac{2\Sigma}{J(\vartheta, \Sigma)^{2}} \int_\R |\Phi'(x)|^2 \mu(x, \vartheta, \Sigma, V) \, dx,
\end{aligned}
```
where
```math
\begin{aligned}
  J(\vartheta, \Sigma) = \| \partial_\vartheta \mathscr{C}_{\vartheta} \|_{L^2(\varphi)}, \quad \Phi'(x) = \frac{1}{\Sigma \mu(x, \vartheta, \Sigma, V)} \int_{-\infty}^x h(z) \mu(x, \vartheta, \Sigma, V) \, dz, \quad x \in \R.
\end{aligned}
```
Here ``\mu(x, \vartheta, \Sigma, V)`` corresponds to [`μ`](@ref) with ``V(x) = x^4/4-x^2/2``. In this case it holds
```math
\begin{aligned}
  &J(\vartheta, \Sigma) = \int_\R (\partial_\vartheta \mu(\vartheta, \Sigma, V) \ast k_\beta)(x) \, \partial_\vartheta \mu(x, \vartheta, \Sigma, V) \, dx, \\[0.5cm]
  &h(z) = (\partial_\vartheta \mu(\vartheta, \Sigma, V) \ast k_\beta)(z) - \int_\R (\partial_\vartheta \mu(\vartheta, \Sigma, V) \ast k_\beta)(x) \, \mu(x, \vartheta, \Sigma, V) \, dx.
\end{aligned}
```

!!! warning 
    The running times for a single evaluation can take some time due to the involved integrations, but it usually needs
    to be computed only once.

---
# Arguments
- `ϑ::Real`:                    positive drift coefficient ``\vartheta``.
- `Σ::Real`:                    positive diffusion coefficient ``\Sigma``.

---
# Examples
```julia-repl
julia> Σ_∞_QrP(1.2, 0.6)
```

---
See also [`μ`](@ref), [`k`](@ref), [`Σ_∞_QdP`](@ref).
"""
function Σ_∞_QrP(ϑ, Σ)
  τ²_QrP(ϑ, Σ)/J_fisher_QrP(ϑ, Σ)^2
end