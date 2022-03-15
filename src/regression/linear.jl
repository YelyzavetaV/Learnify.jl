module Linear

using Statistics: mean

include("../types.jl")
include("enet.jl")

using .Types: AbstractModel
using .Enet: enet!

function preprocess!(X::M, y::V, intercept::Bool=true) where {M<:AbstractMatrix, V<:AbstractVector}
    X̄ = zeros(eltype(X), size(X, 2))
    ȳ = zeros(eltype(y), size(y, 1))
    if intercept
        X̄ = mean(X, dims=1)
        ȳ = mean(y)
        X .-= X̄
        y .-= ȳ
    end
    return X̄, ȳ
end

abstract type LinearModel <: AbstractModel end

"""
    ElasticNet(
        X::Matrix{F},
        y::Vector{F};
        α=1.0,
        β=0.5,
        intercept::Bool=true,
        itₘ::Integer=1000,
        ϵ=1e-5,
    )

Elastic Net regression model that solves the optimization problem
    min[R(θ)] = min[1/(2M)‖y - Xθ‖² + αβ‖θ‖₁ + (1/2)α(1 - β)‖θ‖²],
where X and y are the training data with M samples and N features, and ‖⋅‖₁ and ‖⋅‖
denote l₁ and l₂ norms, respectively.
If `intercept` is true, training data will be centered.
`itₘ` is a maximal number of iterations and `ϵ` is a convergence tolerance for the
duality gap.

# References
[1] J. Friedman et al. "Regularization Paths for Generalized Linear Models via Coordinate
    Descent". J. Stat. Softw. 33(1):1-22, 2010.
"""
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
        intercept::Bool=true,
        itₘ::Integer=1000,
        ϵ=1e-5,
    ) where {F}
        X̄, ȳ = preprocess!(X, y, intercept)
        θ = zeros(F, size(X, 2))

        Nᵢ, η = enet!(θ, X, y, α, β; itₘ, ϵ)

        I = (ȳ .- X̄ * θ)[1]

        new{F}(I, θ, Nᵢ, η)
    end

end

function (m::ElasticNet)(pts::AbstractVecOrMat)
    pts * m.θ .+ m.I
end

"""
"""
function Lasso(
    X::Matrix{F},
    y::Vector{F};
    α=1.0,
    intercept::Bool=true,
    itₘ::Integer=1000,
    ϵ=1e-5,
) where {F}
    ElasticNet(X, y; α=α, β=1.0, intercept=intercept, tₘ=itₘ, ϵ=ϵ)
end

mutable struct Logistic{F} <: LinearModel
end

end
