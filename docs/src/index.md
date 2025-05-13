# Home

This is the documentation website for MDEforM.jl (MDE for Multiscale). For detailed theoretical and numerical information on the minimum distance estimation (MDE) method please refer to the accompanying article XXX.

In order to use the package, clone the [GitHub repository](https://github.com/UQatKIT/MDEforM) to your local machine, navigate through the Terminal to the root directory containing the
Project.toml file, start Julia and activate the project in the Terminal via
```
$ julia --project=.
```
Then hit ] and update and instantiate the packages in the project
```julia-repl
(MDEforM) pkg> instantiate
(MDEforM) pkg> update
```
You may now use the package's functionality
```julia-repl
julia> using MDEforM
```
The functions listed under [Index](@ref Index_index) are exported by the package and thorough documentation of these functions 
can be found in the list of [Contents](@ref Contents_index) at the bottom of this page.

## Toy Example

We want to provide a small example to illustrate the main functionality of the package. We start from the following 2-dimensional fast-slow system of stochastic differential equations (SDE)
```math
\begin{aligned}
  dX_ϵ(t) = -α V'(X_ϵ(t)) - \frac{1}{ϵ} p'\left( \frac{X_ϵ(t)}{ϵ} \right) dt  + \sqrt{2 σ} dU(t), \quad &X_ϵ(0) = x_0, \\
  dY_ϵ(t) = -\frac{α}{ϵ} V'(X_ϵ(t)) - \frac{1}{ϵ^2} p'\left( Y_ϵ(t) \right) dt  + \sqrt{\frac{2 σ}{ϵ^2}} dU(t), \quad &Y_ϵ(0) = y_0.
\end{aligned}
```
Here, ``V`` is a so-called large-scale potential, ``p`` is a ``2\pi``-periodic function, ``U`` is a standard Brownian motion, and the parameters ``\alpha, \sigma, \epsilon`` are strictly positive. An examplatory function constellation for ``V`` and ``p`` is provided by the function [`NLDO`](@ref).

Here is a self-explanatory graphic of ``V`` and ``V + p``:

```@setup potential_graphic
using CairoMakie

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
```

```@example potential_graphic
drift_fig   # hide
```

When ``\epsilon`` converges to zero, then, by homogenization theory, the process ``X_\epsilon`` converges weakly in ``C([0, T]; \R)`` to the process ``X`` solving the SDE
```math
\begin{aligned}
  dX(t) = -α K V'(X(t)) dt  + \sqrt{2 σ K} dW(t), \quad &X(0) = X_0,
\end{aligned}
```
where ``K>0`` is a corrective constant that comes from the cell problem of the homogenization, see also [`K`](@ref). The task is to estimate the parameter ``\vartheta := \alpha K`` 
through data that comes in form of a long trajectory of the process ``X_\epsilon`` with "small" ``\epsilon > 0``. We will use the [`MDE`](@ref) for the estimation. First, we generate synthetic
data with the function [`Langevin`](@ref). The plot of a trajectory, created with [`produce_trajectory`](@ref), looks like the following.
```@example
# quadratic potential V with sine oscillation p
using MDEforM   # hide
import Random   # hide
Random.seed!(1111)  # hide
T = 10
trajectory = Langevin(5.0, 10.0, func_config=LDO(), α=2.0, σ=1.0, ϵ=0.1, T=T)
fig = produce_trajectory(trajectory, T)
```

We now increase the time horizon ``T`` to obtain accurate estimates for ``\vartheta``.
```@example toy_example
using MDEforM   # hide
import Random   # hide
Random.seed!(1111)  # hide
data = Langevin(5.0, 10.0, func_config=LDO(), α=2.0, σ=1.0, ϵ=0.1, T=1000)[1]   # only slow process
nothing # hide
```
Under this configuration the true parameter ``\vartheta`` is given by
```@example
using MDEforM   # hide
ϑ = 2.0*K(LDO()[3], 1.0)
```
We can now use the MDE to estimate ``\vartheta``. Due to parameter identification issues we must also provide the limit diffusion parameter ``\Sigma := \sigma K`` as input.
```@example toy_example
MDE_value = MDE(data, "Langevin", 1.0*K(LDO()[3], 1.0), 10.0)
```

As we can see the MDE is capable of retrieving the true parameter with multiscale data.


## [Contents](@id Contents_index)

```@contents
Pages = ["multiscale_limit_pairs.md", "invariant_densities.md", "MDE_functionals.md", "MDE_optimizers.md", "MDE_asymptotic_variances.md"]
Depth = 1
```

## [Index](@id Index_index)

```@index
```
