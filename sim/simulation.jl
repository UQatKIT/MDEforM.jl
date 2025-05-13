########################################################################################################
########################################################################################################
# Short examplatory code snippet for various parameter simulation studies of the MDE method for 
# multiscale diffusions. The code may has to be changed depending on the limit model of interest.
########################################################################################################
# SDE: Overdamped Langevin diffusion with a fast oscillating part in 1D
# Effective drift function: Quartic potential
########################################################################################################
########################################################################################################
# Jaroslav Borodavka, 09.05.2025

@show Threads.nthreads()

using MDEforM, Dates, Statistics, JLD2
using Base.Threads # start julia and activate project with desired number of threads in terminal, e.g.: $ julia --threads 10 --project=. 

## Monte Carlo parameter study ##

function MDE_single_loop(T, rev, α=2.0, σ=1.0, ϵ=0.1, ϑ_initial=10.0)
    V = NLDO()[1]
    limit_diffusion_parameter = σ*K(NLDO()[3], σ)
    MDE_loop_values = Array{Float64}(undef, rev)

    for i in 1:rev
        data = Langevin(10.0, 10.0/ϵ, func_config=NLDO(), α=α, σ=σ, ϵ=ϵ, T=T)[1]      # true parameter: ϑ=αK(NLDO()[3], σ)
        MDE_loop_values[i] = MDE(data, "Langevin", V, limit_diffusion_parameter, ϑ_initial)
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
jldsave("MDE_NLDO_eps01_1D_values.jld2"; MDE_aver_stdev_values = MDE_aver_stdev_values)

#=
# visualization of robustness

# loading required output data
MDE_aver_stdev_values = load("MDE_NLDO_eps01_1D_values.jld2")["MDE_aver_stdev_values"]

# visualization of robustness

using CairoMakie

limit_drift_parameter = 2.0*K(NLDO()[3], 1.0)        # true parameter: ϑ=αK(NLDO()[3], σ)

MDE_aver_values = MDE_aver_stdev_values[1,:]
MDE_stdev_values = MDE_aver_stdev_values[2,:]
 
val_l = minimum(MDE_aver_values-1.2MDE_stdev_values)
val_u = maximum(MDE_aver_values+1.2MDE_stdev_values)

# create and adjust figure components
robustness_fig = Figure(size=(2560,1440), fontsize = 40)
robustness_ax = Axis(robustness_fig[1, 1],
    # title
    title = L"MDE estimates $\hat{\vartheta}_T \; (X_\epsilon)$ for $\vartheta_0$ when $\epsilon = 0.1$",
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

STD_band = band!(robustness_ax, T_range, MDE_aver_values-MDE_stdev_values, MDE_aver_values+MDE_stdev_values,
                color = (:lightblue, 0.5)
)
MDE_line = lines!(robustness_ax, T_range, MDE_aver_values, linewidth = 3.0)
limit_drift_parameter_line = hlines!(robustness_ax, limit_drift_parameter, color = (:red, 0.8), linewidth = 5.0, linestyle = :dash)

axislegend(robustness_ax,
    [limit_drift_parameter_line, MDE_line, STD_band],
    [L"True drift parameter $\vartheta_0$", L"MDE estimates $\hat{\vartheta}_T \; (X_\epsilon)$", L"$1$ standard deviation band"]
)
robustness_fig
=#