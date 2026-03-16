## Last example in the numerics section of the MDE on the bilinearly coupled fast-slow SDE system with Ornstein-Uhlenbeck limit. ##
using LinearAlgebra
using MDEforM

struct BilinearFastSlow{T<:Real, V<:AbstractVector{T}, M<:AbstractMatrix{T}}
    initial_data::V
    γ::V
    D::M
    σ::M
    α::T
    Σ::T
    c::T
end

# general multiscale system
function simulate(model::BilinearFastSlow, ε::Real=0.1; T::Real=100, dt::Real=1e-3)
    N = convert(Int64, T/dt)
    m = length(model.initial_data)-1

    X = Vector{Float64}(undef, N)
    Y = Array{Float64}(undef, m, N)
    X[1] = model.initial_data[1]
    Y[:,1] = model.initial_data[2:end]

    for i in 1:(N-1)
        dW_x = sqrt(dt)*randn(1)[1]
        dW_y = sqrt(dt)*randn(m)

        X[i+1] = X[i] + ((1/ε) * dot(model.γ, Y[:,i]) - model.α * X[i])dt + model.Σ * dW_x
        Y[:,i+1] = Y[:,i] + ((-1/ε^2) * model.D * Y[:,i] + (model.c/ε) * X[i] * Y[:,i])dt + (1/ε) * model.σ * dW_y
    end

    (X, Y)
end

# general limit process
function simulate(model::BilinearFastSlow; T::Real=100, dt::Real=1e-3)
    N = convert(Int64, T/dt)

    X = Vector{Float64}(undef, N)
    X[1] = model.initial_data[1]

    inv_D = inv(model.D)
    Σ_bar = sqrt(model.Σ^2 + transpose(model.γ) * inv_D * model.σ * transpose(model.σ) * transpose(inv_D) * γ)

    for i in 1:(N-1)
        dW_x = sqrt(dt)*randn(1)[1]

        X[i+1] = X[i] + (-model.α*X[i])dt + Σ_bar * dW_x
    end

    X
end

function homogenized_diffusion(model::BilinearFastSlow)
    inv_D = inv(model.D)
    Σ_bar = model.Σ^2 + transpose(model.γ) * inv_D * model.σ * transpose(model.σ) * transpose(inv_D) * γ
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
c = 1.0             # must be real

model = BilinearFastSlow(xy0, γ, D, σ, α, Σ, c)

#multiscale_system = simulate(model, 0.1, T=5000.0, dt=1e-3)[1]
#limit_system = simulate(model, T=10000.0, dt=1e-3)

#MDEforM.produce_trajectory(multiscale_system[1], 10)
#MDEforM.produce_trajectory(limit_system, 10)
#MDEforM.produce_trajectory(multiscale_system[2][3,:], 10)
#V = LDO()[1]

#mde_res = MDE(multiscale_system, "Fast Chaotic Noise", model.α, 10.0)

@show Threads.nthreads()

using Dates, Statistics, JLD2
using Base.Threads # start julia and activate project with desired number of threads in terminal, e.g.: $ julia --threads 10 --project=. 

## Monte Carlo parameter study ##

function MDE_single_loop(T, rev, ϑ_initial=10.0)
    MDE_loop_values = Array{Float64}(undef, rev)

    for i in 1:rev
        data = simulate(model, 0.1, T=T, dt=1e-3)[1]
        MDE_loop_values[i] = MDE(data, "Fast Chaotic Noise", model.α, ϑ_initial, verbose=false)
    end

    [mean(MDE_loop_values), std(MDE_loop_values)]
end

# simulation via multithreading

T_range = range(100, 2000, 20)      # different time horizons of the process

MDE_aver_stdev_values = Array{Float64}(undef, 2, length(T_range))

time_stamp_start = Dates.format(now(), "H:MM:SS")
@info "∇ $(time_stamp_start) - Start of simulation runs."

# Thread-safe parallel loop
@threads for i in eachindex(1:length(T_range))
    MDE_aver_stdev_values[:,i] = MDE_single_loop(T_range[i], 1000)
    time_stamp_loop = Dates.format(now(), "H:MM:SS")
    @info "∇ $(time_stamp_loop) - T = $(T_range[i]) complete."
end

time_stamp_end = Dates.format(now(), "H:MM:SS")
@info "∇ $(time_stamp_end) - End of simulation runs."

@show MDE_aver_stdev_values


# saving output data
#jldsave("MDE_BFS_eps01_values.jld2"; MDE_aver_stdev_values = MDE_aver_stdev_values)

# visualization of robustness

# loading required output data
MDE_aver_stdev_values = load("MDE_BFS_eps01_values.jld2")["MDE_aver_stdev_values"]

# visualization of robustness

using CairoMakie

MDE_aver_values = MDE_aver_stdev_values[1,:]
MDE_stdev_values = MDE_aver_stdev_values[2,:]
 
val_l = minimum(MDE_aver_values-1.2MDE_stdev_values)
val_u = maximum(MDE_aver_values+1.2MDE_stdev_values)

# create and adjust figure components
robustness_fig = Figure(size=(2560,1440), fontsize = 40)
robustness_ax = Axis(robustness_fig[1, 1],
    # title
    title = L"MDE estimates $\hat{\vartheta}_T \; (X^\varepsilon)$ for $\vartheta_0$ when $\varepsilon = 0.1$",
    titlegap = 25,
    titlesize = 50,
    # x-axis
    xlabel = L"T",
    xticks = LinearTicks(5),
    # y-axis
    yticks = LinearTicks(10),
)
Makie.xlims!(robustness_ax, T_range[begin], T_range[end]), Makie.ylims!(robustness_ax, val_l, val_u)
colsize!(robustness_fig.layout, 1, Aspect(1, 1.8))

STD_band = band!(robustness_ax, T_range, MDE_aver_values-MDE_stdev_values, MDE_aver_values+MDE_stdev_values, color = (:green, 0.125))

MDE_line = lines!(robustness_ax, T_range, MDE_aver_values, linewidth = 3.0, color = (:green, 1.0))

limit_diffusion_parameter_line = hlines!(robustness_ax, homogenized_diffusion(model), color = (:red, 0.8), linewidth = 5.0, linestyle = :dash)

axislegend(robustness_ax,
    [limit_diffusion_parameter_line, MDE_line, STD_band],
    [L"True diffusion parameter $\vartheta_0$", L"MDE estimates $\hat{\vartheta}_T \; (X^\varepsilon)$", L"$1$ standard deviation band"]
)
robustness_fig
save("MDE_BFS_robustness_eps01.pdf", robustness_fig)