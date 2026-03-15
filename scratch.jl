# just a helper file for repeated testing during development, instead of using the REPL for large code chunks

using Revise    # for continuous testing without restarting Julia after each change to the main code
using MDEforM


# general multiscale system
function BilinearFastSlow(xy0::Vector{<:Real}; γ::Vector{<:Real}, α::Real, Σ::Real, D::Array{<:Real, 2}, ϵ::Real=0.1, T::Real=100, dt::Real=1e-3)
  
end
