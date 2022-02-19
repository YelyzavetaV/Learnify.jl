module Linear

using Statistics: mean

include("../types.jl")
include("enet.jl")

using .Types: AbstractModel
using .Enet: enet!

function preprocess!(
    X::M, y::V, center::Bool=true, normalize::Bool=false
) where {M<:AbstractMatrix, V<:AbstractVector}
    X̄ = zeros(eltype(X), size(X, 2))
    ȳ = zeros(eltype(y), size(y, 1))
    if center
        X̄ = mean(X, dims=1)
        ȳ = mean(y)
        X .-= X̄
        y .-= ȳ
    end

    if normalize
    end

    return X̄, ȳ
end

abstract type LinearModel <: AbstractModel end

mutable struct ElasticNet{F} <: LinearModel
    I::F
    θ::Vector{F}
    Nᵢ::Integer
    η

    function ElasticNet(
        X::Matrix{F},
        y::Vector{F};
        α=1.0,
        β=0.5,
        center::Bool=true,
        normalize::Bool=false,
        itₘ::Integer=1000,
        ϵ=1e-5,
    ) where {F}
        X̄, ȳ = preprocess!(X, y, center, normalize)
        θ = zeros(F, size(X, 2))

        Nᵢ, η = enet!(θ, X, y, α, β; itₘ, ϵ)

        I = (ȳ .- X̄ * θ)[1]

        new{F}(I, θ, Nᵢ, η)
    end

end

function (m::ElasticNet)(pts::AbstractVecOrMat)
    pts * m.θ .+ m.I
end

end
